---
status: closed-negative
verdict: Porting the engine to Vulkan across four phases — the journal, not the rules. It shipped and decoded, then was RETIRED 2026-08-06 as an unfinished port: 16 of 29 kernels, 6 of its own 36 mode-matrix cells decoding (of 72; 36 per backend), ~1.9x slower, no DeepSeek-V4 path. Code at tag archive/vulkan-backend-hb16; the inventory and the shader rules are in vulkan-kernels.md.
---

# rivoli — Vulkan backend (retired)

> **CORRECTED 2026-08-06.** This was `status: closed-shipped` and its verdict said
> "Shipped and decoding". Both were true when written and neither is now: **the Vulkan
> backend was deleted from the tree on 2026-08-06** and the `vulkan` feature no longer
> exists. The status is `closed-negative` because the code is gone, not because the port
> failed at what it attempted — it did decode, it did reach 97% fetch/compute overlap, and
> those measurements stand.
>
> What changed is the judgement, not the data. Classified by the user as **an unfinished
> port, not a feature**: 16 of 29 kernels, 6 of its own 36 `tests/mode-matrix.sh` cells decoding (of 72; 36 per backend)
> against 30 refusing, ~1.9x slower on the single configuration it supported, and no
> DeepSeek-V4 decode path at all — while every V4 launcher signature change cost a parallel
> edit to a backend that could not use it.
>
> Everything is preserved at the tag **`archive/vulkan-backend-hb16`**. Note that the
> earlier tag `archive/vulkan-backend` points one commit further back and **predates the
> HB=16 work** (`c434de3`, +54/-18 across `vk.rs` and `mla_latent_attend.comp`); prefer the
> `-hb16` tag.
>
> The companion [`vulkan-kernels.md`](vulkan-kernels.md) moved here from `reference/` the
> same day and holds the inventory, the numerics/index-width rules and the two OPEN fp8-dot
> gaps — the parts worth reading if these kernels are ever ported to a third API.

> **NAV — 117 KB, a port journal across four phases. Do not read whole.**
> Current capability is one section: **"Kernel inventory — port 16 of 29"**. That gives
> what runs, what is deferred and why, and (2026-07-31) the six PORTED-but-single-row
> kernels that make speculative decode ROCm-only. `grep -n "^## " docs/investigations/vulkan-port.md` for the
> rest; the phase sections are how it got here, including two conclusions it later
> falsified (Vulkan CAN overlap; timeline waits DO work).
> **One line:** decodes `--mode int3-vq --attn dense`, three queues, ~1.9× slower than ROCm
> and all of it MoE kernel throughput.
>
> **CORRECTED 2026-08-01.** This line also said "97% fetch hidden". **Do not quote that
> number** — `fetch_hidden_pct` was a broken quotient (see the correction ~40 lines below),
> and on 2026-08-01 the metric was deleted from the engine outright rather than kept in
> corrected form, along with `exposed_fetch_ms`, `rivoli_fetch_hidden_pct` and the
> `split/exposed-fetch` series. The Vulkan overlap finding does **not** depend on it: it
> rests on increment 1 vs increment 2 at fixed everything-else. Every `fetch_hidden_pct`
> figure below is kept as recorded, because what it *ruled out* — that a green correctness
> check certifies an overlap invariant — is this journal's most reused lesson.

Status: **RUNNABLE, AND IT IMPLEMENTS THE DESIGN. Phase 4 complete (2026-07-30):
`--features vulkan` builds and decodes `--mode int3-vq --attn dense` over THREE QUEUES with
the fetch↔compute overlap intact.** 16 kernels + `flag_nonfinite` + `fill_u32`, 46/46 green
under the validation layer; the deliberate straddle break verified red-then-green; GPU-AV
confirmed live by re-running the probe coverage matrix rather than by a clean pass.

**Fetch hidden: 97%**, measured with real timestamp query pools, against ROCm's 96% on the
same matched run — and against 0% in increment 1, which was an arithmetic artifact of a
stubbed `elapsed_ms` rather than a measurement.

> **CORRECTION 2026-08-01 — the 96/97% figures are inflated; the finding they support is
> not.** `fetch_hidden_pct` was `1 − (moe_wall − compute_gpu)/fetch_wall`, and `compute_gpu`
> is a bracket that *contains* the stalls it was being used to rule out, so it read ~97% on
> any configuration whatsoever — including `--direct-vmm-dma`, which printed 99% while
> decoding at half speed. Recomputed against a measured all-resident counterfactual
> (`docs/reference/architecture.md` §3), ROCm is ~22%. The Vulkan number has not been re-measured.
>
> **What survives:** increment 1 ran every stream on one queue and genuinely hid nothing,
> and increment 2's three queues genuinely overlap — that ordering was never in doubt and
> is what the increment was for. What does not survive is reading 96% as "the fetch is
> nearly free". It is not: ROCm is disk-bound, §3.

Vulkan is still **1.87x slower** end to end (1.46 vs 2.73 tok/s), and the per-phase GPU spans
now say why: **279 ms of the 320 ms/token gap is the MoE kernels themselves**, 2.1x slower
than the HIP originals. The overlap invariant is upheld; what remains is kernel throughput,
which is a different problem with a different fix. See "Increment 2: measured" — including
why 97% hidden is not a permanent property, since the fetch only just fits behind the current
(slow) compute.

On matched runs the two backends produced **identical token IDs** at K = 2. Gate A's **K is
still unmeasured**: agreement at K = 2 is a floor, not a result. docs/reference/modes.md numbers still come
after a kernel-throughput pass, not after this one.

The acceptance gate is **A+C+D** (see below); byte-identical token IDs are unattainable by
construction and that is measured, not suspected. Sections below describing work as
"planned" predate the merge — the staging table and the 2d pre-flight are current.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the engine it plugs into.

## Goal

A second compute backend behind `--features vulkan`, mirroring `rocm`: same engine,
same artifact, same numerics, no ROCm/HIP toolchain required to build or run. The
motivation is portability (any Vulkan 1.3 driver — RADV, AMDVLK, and in principle
Intel/NVIDIA), not speed. **HIP stays the default and the performance reference.**

**One backend per build — ROCm *or* Vulkan, chosen at build time. There is no hybrid
mode.** A build contains exactly one implementation; no runtime selection, and no
splitting work across backends within a run (no "MoE on Vulkan, attention on HIP").
This is what makes the portability claim real: a Vulkan build links no HIP and needs
no hipcc, which a both-backends binary could not do.

Non-goals: replacing the HIP path; a graphics pipeline; `wgpu`/`vulkano` (see
"Crate choice"); a `--backend` runtime flag (see "Backend selection").

## Crate choice: `ash`

`ash` is raw Vulkan bindings — zero-cost, no runtime, no allocator, no object model.
That is exactly the shape this codebase already has: every kernel sits behind a
`rivoli_*` C launcher and Rust unsafety funnels through the thin waist in `hip.rs`.

- **`vulkano`** brings its own memory allocator and an `Arc`-heavy object model. The
  engine's core value is hand-placed memory (the two-ended byte `Arena`, pinned
  staging, a policy that evicts by byte range) — vulkano would fight precisely the
  part that is worth keeping.
- **`wgpu`** abstracts away explicit memory placement and queue control, and limits
  subgroup access. Wrong tool for a batch-1, bandwidth-tuned decoder.

`ash` is actively maintained but releases slowly (0.38.0+1.3.281, April 2024, tracking
Vulkan 1.3.281). Pin it; the surface used here is core 1.2/1.3 and stable.

## The contract to reimplement

Three modules define the whole backend boundary. Nothing above them should change.

| Module | What it provides | Vulkan equivalent |
|---|---|---|
| `src/hip.rs` | ~28 `launch_*` fns over 29 compute kernels | pipeline dispatch + push constants |
| `src/device.rs` | `DeviceBuf`, `DeviceTier`, `VmmBuf`, `mem_info` | `VkBuffer` + `VkDeviceMemory` suballocation |
| `src/gpustream.rs` | `HipStream`, `HipEvent`, `Signal` | queue + timeline semaphore + query pool |

Mechanical mappings:

| HIP | Vulkan |
|---|---|
| `hipStream_t` | `VkQueue` + per-submit `VkCommandBuffer` |
| `hipEvent_t` elapsed | `VK_QUERY_TYPE_TIMESTAMP` query pool |
| `hipLaunchHostFunc` | **no equivalent** — see "Signal bridge" |
| `hipMalloc` / `rivoli_vmm_alloc` | `vkAllocateMemory` + offset-bound `VkBuffer` |
| `hipHostMalloc` (bounce arena) | device-local **copy pool** — see "Memory: copy, don't import" |
| `hipMemcpyDtoD` (arena compaction) | `vkCmdCopyBuffer` |
| `hipMemcpyHtoDAsync` | `vkCmdCopyBuffer` on a transfer queue |
| kernel raw pointer args | buffer device addresses (`VK_KHR_buffer_device_address`) |
| `__shared__` | `shared` |
| `__shfl_down(v, o, 32)` | `subgroupShuffleDown(v, o)` |

### Required device features

> **MOVED 2026-08-01 to [`vulkan-kernels.md`](vulkan-kernels.md),
> "Device requirements".** This list is what the backend demands of a device *today* — live
> reference, not a decision the port once made — and twelve shaders cite it. It was the
> largest piece of live content still on the closed shelf. The requirement that generated
> the most shader comments, `VK_KHR_8bit_storage` deliberately NOT required, went with it.
> The port's original reasoning is preserved by this file's history; only the standing
> requirement moved, because a duplicate is how two copies drift.

## Four things that are not mechanical

### Host pointer ≠ device address

The single biggest structural difference between the backends, and the one the
mapping table above quietly assumes away. Under HIP unified addressing the host
pointer and the device pointer are the SAME NUMBER, and the engine leans on it:
`DeviceTier::reserve` returns one pointer that `pin.rs` uses for both jobs at once.

```rust
// src/pin.rs:190 (place_f32), and the same shape at 211-212, 234-235, 270, 597
let dst = tier.reserve(bytes.len())?;
unsafe { copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len()) };  // host write
Ok(dst as *const f32)                                              // device pointer
```

Same conflation at `pin.rs:717-718`, where `VmmBuf::ptr_mut()` becomes
`ArenaPool.base` — simultaneously the io_uring O_DIRECT DMA target (a host address)
and the base every expert descriptor's six device pointers are computed from
(`pin.rs:427`).

Under Vulkan these are two unrelated numbers. A mapped `VkDeviceMemory` pointer and
the buffer's device address have no fixed relationship, and
`VK_EXT_external_memory_host` would not make them equal either (and is rejected above
on measured performance grounds regardless).

**Resolution (implemented): `DeviceTier::place(&[u8]) -> *mut u8` reserves and fills
in one call and returns the DEVICE pointer; `reserve` is private.** The tier owns both
bases, so the translation happens inside the type that knows them, once per placement
at startup — never per operation, never on the hot path. Fusing the two also means a
caller cannot obtain a device pointer that has not been filled, which a separate
`write_at` would have allowed. `VmmBuf` splits the same way when the pool is ported:
`ptr()` is the device base for descriptor arithmetic, `host_mut()` is the DMA target.

The pool half is NOT done. `ArenaPool::ptr` (`pin.rs`) must become two accessors —
descriptors take the device base, `ReadSpec.dst` takes the host base — with both bases
still resolved once at setup so each stays a single `add`. Deferred deliberately: under
`rocm` the two bases are identical, so splitting it now would compile to the same thing
and no test could tell the versions apart. It lands with the Vulkan streaming path.

Two rejected alternatives, for the record:

- *Keep returning the host pointer and add a `tier.dev(host_ptr)` sibling.* Same
  number of call sites touched, but leaves a host pointer typed as a device pointer
  in every intermediate — a bug waiting for a maintainer.
- *A global (host_base, len, dev_base) table translated inside the launchers.* Zero
  change to `pin.rs`, but it puts a lookup on the hot path — six pointers per expert
  × ~9 experts × 78 layers per token in a latency-bound decoder — to hide a
  distinction that is real and should be visible.

`DeviceBuf` needs none of this: it already does an explicit `copy_in_at`/`copy_out`,
so it maps straight onto `vkCmdCopyBuffer` or a mapped write.

### Same seam, different mechanism

Where a backend cannot do a thing the same way, it must still expose the same seam.
Moving work host-side to preserve exactness is a legitimate port, not an admission of
defeat — and keeping the launcher signature identical is what lets `backend.rs` stay a
build-time `pub use` instead of growing a trait.

`rope_interleave` is the worked example. HIP evaluates `pow`/`cos`/`sin` in **f64**;
GLSL has no double transcendentals at all (Float64 gives arithmetic and `floor`, not
these), and f32 in-shader is not good enough for a reason that had to be derived rather
than assumed:

> The relative error in `inv = theta^(-2j/seg)` is **amplified by `pos`**. At j=0,
> inv=1 and ang=pos exactly — but a 1e-7 relative error in `inv` becomes ~`ang`x1e-7
> absolute angle error, i.e. **~1e-3 rad at pos=1e4**. That is a visibly wrong rotation,
> not a rounding difference.

So the f64 in the HIP source is load-bearing, which nobody could tell by reading it.
`launch_rope` therefore evaluates the same f64 expression on the CPU and uploads a
`seg/2`-entry table. The result is **bit-identical** rather than merely close, because
both backends round the same f64 value to f32 — an in-shader f32 scheme would have been
an approximation dressed as a port.

The cost is one small upload per call and a `Cmd::scratch` list for buffers that must
outlive recording but not the sync. The drop ordering there is not arbitrary: they are
taken out under the mutex and dropped *after* releasing it, because `Buf::drop` calls
`Gpu::sync`, which takes the same non-reentrant lock.

Generalisation for the remaining kernels: **if a mechanism cannot be reproduced, move
the work rather than approximating it, and keep the seam.** Ask what the HIP code is
buying with the construct you cannot mirror, and whether the host can buy the same thing.

### Memory: copy, don't import

The obvious design — import the io_uring bounce arena as a `VkBuffer` via
`VK_EXT_external_memory_host` so the shader reads weights in place — is **measured
slower on our exact hardware** (RADV, gfx1151). Colibri built it, then replaced it with
a fixed pre-allocated round-robin pool of device-local buffers:

    SUBMIT ms/batch:  import(ring128) 2.13 | import-slab-shared 8.62 | copy 0.06

Cause: amdgpu re-validates every *live* host-imported (userptr) BO on every
`vkQueueSubmit`, so submit cost scales with live imports rather than referenced ones —
and a ring evicts imports before they recur, so "zero-copy" re-pins (`get_user_pages`
≈ a memcpy) on every use anyway. The copy pool also survives a mid-batch slab reload
that segfaults the import path.

So: `batch_add`-style memcpy into the next pool buffer, GPU reads a private device-local
snapshot. Size the pool like colibri's default (~32 buffers) and tune from there.

### Deviation: the copy pool is NOT built, because we never import

The measurement above is real, and the conclusion drawn from it does not apply to what
this port actually does. The 2.13 ms submit cost comes from amdgpu re-validating live
**userptr** BOs — host pages the driver does not own, handed to it by
`VK_EXT_external_memory_host`. The pool is a workaround for that specific pathology.

`VmmBuf` is an ordinary `VkDeviceMemory` allocation from the
`DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT` heap — which on Strix Halo is the whole
heap — permanently mapped, with `host_mut()` handing out that mapping. The driver
allocated those pages itself, so there is nothing to re-validate and the premise is
absent. io_uring reads `O_DIRECT` straight into the slot; the GPU reads device-local
memory. That is the plan's *intent* — no import, no userptr re-validation, device-local
reads — reached without the pool and without the memcpy the pool would add on hardware
where host RAM already is device memory. Building it anyway would be cargo-culting a
fix for a problem the design avoids.

Colibri's other note, that the pool "survives a mid-batch slab reload that segfaults the
import path", is also import-specific: a Vulkan-owned allocation is not reloaded under
us. The residual concern — a DMA landing in a slot a kernel is still reading — is real
but orthogonal, and already owned by the `inflight` guard and the rule that every miss
slot in a batch is allocated before any read is issued. That is scheduling, not
buffering; a pool would hide it rather than fix it.

**The measurement that overturns this:** if per-submit cost is ever observed scaling
with the number of LIVE buffers rather than the number referenced by the submission,
the userptr pathology has reappeared by another route and the pool comes back. That is
the same signature colibri measured, and it is what to watch for.

**Two premises this makes load-bearing, both now runtime assertions rather than prose:**

- **4096-byte alignment** of the mapped base, or `O_DIRECT` fails `EINVAL` at the first
  read, inside the reaper, on someone else's machine. `vkMapMemory` guarantees only
  `minMemoryMapAlignment`, which is page-sized everywhere we have looked and is not
  required to be. Checked in `VmmBuf::new` (`vk::O_DIRECT_ALIGN`). The stride half of
  the requirement is `VQ_ALIGN`, enforced by the arena.
- **`HOST_COHERENT`** on the selected memory type, or host writes need an explicit
  `vkFlushMappedMemoryRanges` that this backend does not perform — and the failure mode
  is stale data read as valid: no fault, no validation message, wrong numbers.
  `Buf::new` refuses to allocate without it and says why.

### Determinism

### Determinism

Greedy decode must stay reproducible, which is why every reduction in
`kernels/common.hpp` uses a fixed `__shfl_down` ladder rather than an atomic or
library reduce. **Do not port `wave_sum` to `subgroupAdd`** — its summation order is
implementation-defined. Use `subgroupShuffleDown` and keep the same halving ladder;
`GL_KHR_shader_subgroup_shuffle_relative` maps 1:1. Same rule for the LDS combine in
`gemv_fp8_splitk` and `mla_attend_combine`: fixed loop order, not an atomic.

### Signal bridge

`src/gpustream.rs` turns a GPU completion into a futures `Waker` via
`hipLaunchHostFunc` (~19 µs/signal). Vulkan has no host-callback-on-queue. Replacement:
one waiter thread blocking in `vkWaitSemaphores` on a timeline semaphore, resolving the
`Signal` for each value it observes. The `Signal` API (`pending`/`ready`/`arm_on`/
`resolve`) stays as-is, so `asyncfetch.rs` and the expert stream in `gpu.rs` do not
change.

**MEASURED: 19.28 µs/signal** (200 iterations, `timeline_signal_resolves_and_latency`),
against the HIP path's ~19 µs. A thread wakeup neither beats `hipLaunchHostFunc` nor
regresses against it — the two are within noise, so the contingency of batching several
completions per semaphore value is not needed, and the async overlap costs the same on
either backend. Do not treat this as headroom: at ~9 signals per layer over 78 layers
it is ~13 ms/token on both paths, which is a real cost that simply is not a REASON to
prefer HIP.

One caveat on the number: it was measured with the rest of the suite running on
parallel threads against the same queue. That makes it an upper bound under contention
rather than a quiet-machine best case, which is the more useful figure here anyway.

## The toolchain rewrites float arithmetic. Assume nothing compiles literally.

Read this before any other numerics claim here, because it invalidates the shape of
argument the rest of them are made in.

`append_kv.comp` wrote `a / 448.0`, matching `fwd.hip` character for character. The
module `glslc --target-env=vulkan1.3 -O` produced contained **no division**:

    %289 = OpFMul %float %645 %float_0_00223214296        // = fl(1/448)

Without `-O` it is `OpFDiv %float %288 %float_448`. `hipcc` gets `-O3` and no
`-ffast-math`, so the HIP side keeps a true IEEE divide, and `x * fl(1/448)` differs
from `x / 448` by 1 ULP on **55.1% of inputs** (measured, 200k samples). Every
KV-cache block scale diverged between the backends, and any `lc8` byte whose quotient
sat near an e4m3 rounding midpoint could differ too.

**No test could have caught it, and none was going to.** The `lscale` oracle uses a
1e-6 relative tolerance (~8 ULP). The byte-exact `lc8` comparison is guarded by
`assert_quantization_unambiguous(..., 8, ...)` — and *the margin that makes that
comparison stable across drivers is the same margin that hides a 1-ULP scale shift*. A
precondition built to protect a comparison turned out to be load-bearing in two
directions at once. It took reading disassembly.

**The consequence for the determinism story.** This port's reproducibility argument —
the fixed `subgroupShuffleDown` ladder, `subgroupAdd` banned by `build.rs` — quietly
assumed the toolchain does not rewrite float arithmetic. That assumption is now
*measured false*. The `subgroupAdd` rule is necessary and **not sufficient**: it
constrains what the author writes, not what the optimiser emits. Every claim in this
document of the form "the same expression compiles to the same operation" has to be
read in that light, and verified against disassembly rather than against source.

Two mitigations, both mechanical:

- **Pass constants the optimiser must not fold as runtime operands.** An operand it
  cannot see is one it cannot fold. `append_kv` takes `E4M3_MAX` through its push
  constant, sourced from `math.rs` so the value still has one definition.
- **`build.rs` rejects any `OpFMul` by a float constant with a non-zero mantissa.**
  Powers of two are exact and common in the e4m3/bf16 paths; anything else is either an
  invented reciprocal or an author-written scale that deserves an argument. Verified to
  fire by reinstating the literal.

When porting the remaining kernels, **disassemble anything whose bit-exactness matters**
rather than trusting that the source says what it means.

**And the toolchain is only one of the two sources.** The other is the porter reaching
for the idiomatic spelling. `1.0/sqrt(z)` and `inversesqrt(z)` compute different numbers
— Vulkan specifies the latter to 2 ULP as a single operation, HIP does a
correctly-rounded `sqrt` then a correctly-rounded divide — and `inversesqrt` is exactly
what a reviewer would "tidy" the explicit form into. The reciprocal guard cannot see
this: it inspects an `OpFMul` constant, and a function substitution has none. `build.rs`
now rejects the GLSL.std.450 `InverseSqrt` opcode by name, as a denylist that grows each
time another such pair is identified. A one-entry denylist that fires exactly beats no
guard.

### Bit-exactness with `math.rs`

> **MOVED 2026-08-01 to [`vulkan-kernels.md`](vulkan-kernels.md),
> "Numerics that must stay bit-exact" and "Index width".** Both are standing obligations on
> anyone editing a shader, not port history: the NaN divergence between `f2bf16` and
> `half::bf16::from_f32` still binds, and the index ceiling is cited by five shaders that
> each record their own peak. `common.glsl` cited this section as "Numerics" and had no
> anchor to land on — that heading now exists there.

## Backend selection

Add a `src/backend.rs` that re-exports one implementation:

```rust
#[cfg(all(feature = "rocm", not(feature = "vulkan")))]
pub use crate::hip::*;
#[cfg(all(feature = "vulkan", not(feature = "rocm")))]
pub use crate::vk::*;
#[cfg(all(feature = "rocm", feature = "vulkan"))]
compile_error!("features `rocm` and `vulkan` are mutually exclusive");
```

Then `gpu.rs` and `pin.rs` import `crate::backend::*` instead of `crate::hip::*`. No
trait, no dynamic dispatch, no runtime cost — the two backends are never live at once.
A trait would buy nothing here and would force every launcher signature through a
vtable on the hot path.

**Rejected: a runtime `--backend rocm|vulkan` flag.** It reads natural next to the
existing `--mode`/`--attn` flags, but one binary containing both would have to link
`libamdhip64` and compile the HIP kernels unconditionally — so a Vulkan-only machine
could not build it, which is the entire point of the feature. It would also put an
enum match in front of every `launch_*` on the decode hot path. Build-time selection
costs one `cargo build` flag and nothing else.

## Acceptance gate

The port has an unusually good test story: `tests/kernel.rs` already validates every
kernel against CPU oracles in `math.rs`/`quant.rs`. **The Vulkan backend is done when
the oracles for the 16 ported kernels pass**, plus a coherent 32-token decode on a
partial artifact (the `convert --layers 1` stub path used for the HIP end-to-end
bring-up).

### THE GATE BELOW CAN NEVER PASS. A replacement is proposed, and it is the USER'S CALL.

The bar written in the next section — a byte-identical token-ID sequence against the HIP
path — was set when `exp` was believed to be a tolerable unknown. It is now **unattainable
by construction**, and that is measured rather than suspected:

- `tests/xbackend.rs` compares the two backends directly on `swiglu`, the smallest kernel
  whose result depends on `exp`. **1463 of 4096 outputs differ**, in the band where `exp`'s
  low bits reach the result.
- Reproducing hipcc's `expf` argument reduction instruction-for-instruction in GLSL left
  that number **unchanged at 1463**, moving only 12 values. The residue lives in the
  hardware `v_exp_f32` each toolchain calls, which GLSL source cannot reach.
- All three constructive steps of the exactness strategy have been ATTEMPTED on `exp` and
  each fails for a stated reason. It is not that nobody has tried.

So the logits differ in their low bits on every layer, and over 78 layers and a
154,880-way argmax a divergence is a matter of when, not whether. **A gate that cannot pass
is not a strict gate; it is an absent one**, because the first time it fails everyone will
reach for a reason to discount it.

**Deciding what replaces it is a product decision, not an engineering one** — it defines
what "the Vulkan backend is acceptable" means — so what follows is a proposal to take to
the user, not a change already made.

#### The proposal: four parts, because no one of them is sufficient

**A. Token-ID agreement for K tokens, where K is MEASURED and REPORTED, not chosen.**
Fixed prompt, greedy decode, same artifact, `--mode`, `--max-mem` and `--cache-policy` on
both backends; report the index of the first differing token.
*Catches:* every gross defect. A wrong index, a dropped row, a mis-ordered reduction, a
bad expert — all of these diverge within a handful of tokens. This is by far the most
sensitive instrument available and it stays that way.
*Cannot catch:* nothing, once K is large. The number is a floor on agreement, not a
pass/fail, and a low-bit `exp` difference will eventually end it for benign reasons. **The
failure mode to guard against is treating a large K as a licence to stop looking.**
*Suggested shape:* record K per release; a K that suddenly collapses from hundreds to
single digits is the alarm, not the absolute value.

**B. Perplexity equivalence within a pre-registered bound.** Teacher-forced on a fixed
corpus, paired per-token NLL, reported as a mean difference with a 95% CI — the machinery
`bin/ppl` already has, and the same verdict vocabulary already in `docs/measurement/benchmarks.md`
(PASS / FAIL / COST ESTABLISHED / INCONCLUSIVE).
*Catches:* systematic accuracy loss that a short token-ID run would miss entirely, and it
is the only part that speaks to output QUALITY rather than agreement.
*Cannot catch:* a rare catastrophic path the corpus never exercises; and it is
underpowered at small n — `docs/measurement/benchmarks.md` already records three of four cells being unable
to resolve a 1% question at 762 tokens. Budget the corpus accordingly or the verdict will
be INCONCLUSIVE by construction.

**C. Throughput and overlap against the ROCm baseline.** tok/s and fetch-hidden %, at
matched `--max-mem` and `--cache-policy`.
*Catches:* the failure A and B are both structurally blind to — a correct backend whose
concurrency has collapsed. A fetch-serialised build computes identical numbers, so it
passes A perfectly and B perfectly while being several times slower.
*Cannot catch:* any correctness property whatsoever.
*Why it must be in the gate rather than in a benchmark report:* the single-queue finding
above means this is the most likely way integration goes wrong, and it is the only part of
the gate that would notice.

**D. Per-kernel bit-exactness where it is attainable, as a standing regression net.**
Already built for `mla_value_fp8`, `mla_absorb_fp8`, and the `exp`-free half of the
attention (`err = 0.000e0`).
*Catches:* ordering, grouping and contraction changes — which A, B and C would all absorb
silently, since each is a low-bit effect that only sometimes reaches an argmax.
*Cannot catch:* anything downstream of `exp`, by definition. That is exactly the region
the other three cover.

#### DECIDED BY THE USER: the gate is A + C + D. B is OUT.

**Accept on token-ID agreement for K tokens (K measured and reported) + throughput/overlap
within bound against the ROCm baseline + per-kernel bit-exactness green where attainable.**
Perplexity equivalence is **not** a merge condition.

C is the pass/fail criterion, D runs on every commit, and A is a reported number rather
than a threshold.

##### The blind spot this buys, stated as a limitation and not as a caveat

**The accepted gate cannot see quality drift that preserves the argmax for K tokens and
degrades afterwards.** B was the only part that spoke to output quality rather than to
agreement or speed; without it, a backend that decodes identically for K tokens and then
diverges into subtly worse text passes everything. A would report a healthy K, C would be
green, D is bit-exactness on kernels that never reach `exp`.

That is a real hole and it is being accepted knowingly, not overlooked. Two consequences
follow, and they are obligations rather than suggestions:

- **`bin/ppl` stays built, documented and runnable as a NON-GATING diagnostic.** It stops
  being a merge condition; it does not stop being available. The first person who suspects
  quality drift should find an instrument already in the tree — with the paired-NLL
  machinery, the CI, and the four-verdict vocabulary already in `docs/measurement/benchmarks.md` — rather
  than an argument about whether to build one. Removing it because "it is not in the gate"
  would convert an accepted, bounded blind spot into an unbounded one.
- **K now carries the weight B used to.** It is the only part of the gate with any
  sensitivity to numerical drift at all, so it must be REPORTED PROMINENTLY and its
  baseline RECORDED THE FIRST TIME IT IS MEASURED. A K of several hundred means nothing to
  a reader with no prior; a later collapse from hundreds to single digits is only visible
  as a regression if the earlier value is written down. Record it here, next to the run
  that produced it, and treat a large K as evidence about that run rather than as licence
  to stop looking.

The honest summary of the change: **the old gate asked "are the backends identical?" and
the answer is now known to be no. The replacement asks "is the Vulkan backend as good, and
as fast, and does it diverge only where we have shown it must?"** That is weaker, and it is
the strongest question that still has a true answer.

Worth noting this is option (b) from the pre-registered contingency list in the `swiglu`
section — written down before the measurement existed, precisely so the choice could not be
made after seeing the result. It is being taken for the reason it was written.

### The real bar: an identical TOKEN ID sequence against the HIP path

**NOT ATTAINABLE TODAY, and that is the point of writing it down.** The Vulkan backend
cannot run the model — no MoE, no attention, no integration — so nothing here can be
executed until phase 4 puts a whole forward pass on the Vulkan path. A gate nobody *can*
run reads like a gate nobody *has* run, and those are different claims; this one is the
former, and the section says so up front so it is never quietly counted as passed.

Per-kernel oracles are necessary and are not the acceptance test. Each checks a kernel
against a CPU reference *in isolation*, at 1e-3, on the shapes someone chose. None sees
an error that appears only once 78 layers compose, none exercises the shapes and
accumulation lengths the real model actually reaches, and none constrains the sampled
token at all. "Coherent output" is a human judgement that a wrong-but-plausible decode
passes.

**The bar: the same prompt decoded through both backends yields an identical sequence of
TOKEN IDs.**

Compare IDs, not decoded text. Two different ID sequences can decode to the same string —
tokenizers round-trip whitespace and multi-byte boundaries in ways that absorb a
difference — so text equality is the weaker claim, and the weakness is exactly in the
direction that hides a defect. Comparing text happens to be sufficient when a single-ULP
shift is unsurvivable over hundreds of argmaxes; the strict form is what to write down.

Pin every condition or it is not a comparison: **fixed prompt, greedy decode, same
artifact, same `--mode`, same `--max-mem` budget, same cache policy.** The budget and
policy matter as much as the prompt — they determine residency and therefore which
experts are recomputed versus reused, and an unpinned difference there produces a
divergence that has nothing to do with the backend. The same discipline the perf A/B
needed when it matched miss counts to the decimal.

Greedy decode is what makes this a legitimate demand rather than an aspiration: every
token is an argmax over a 154,880-way vocabulary, so one differing logit that crosses a
boundary anywhere in the run changes the ID sequence and the comparison fails loudly. It
subsumes every per-kernel oracle — nothing can be wrong in a way that survives it.

**It is demonstrated achievable on this hardware.** Not on this backend: the perf work's
interleaved A/B produced 256 greedy argmaxes byte-identical between arms across a real
decode. So the property does hold for a correct change here, and a Vulkan run that fails
it has a defect rather than an unrealistic target.

### The incremental form: per-kernel golden bits — BUILT, and not as a hash

**Delivered in tranche 2b, in a stronger form than the design below specified.** The plan
was to record a HASH of a HIP-produced output buffer and assert the Vulkan oracle against
it. What exists instead is a **spec-derived bit-exact CPU oracle**: `mla_value_fp8` and
`mla_absorb_fp8` are compared element-by-element against a Rust reference at
`err = 0.000e0`, exactly, not within a tolerance.

**A hash tells you something changed; a derived oracle tells you what the answer should
be.** That difference is the whole value, and it runs in two directions:

- A hash cannot survive a legitimate change, so the first time an intended edit moves the
  bits, someone regenerates it — and the failure message that says DO NOT REGENERATE is
  competing with the path of least resistance at the exact moment it matters. A derived
  oracle has nothing to regenerate: it is computed from `mla.hip` and `math.rs`, so a
  legitimate change updates it by construction and an illegitimate one still fails.
- A hash is evidence only about the machine that produced it. The oracle is a statement
  about the specification, so it fails on the RIGHT machine too.

Two things had to be true before it worked, and neither was obvious:

1. **The oracle must model FMA contraction** (see the class table above). Modelling the
   arithmetic as the shader source spells it left 10 of 15 outputs wrong in their low bits.
2. **The comparison must be on BYTES.** See the next section — a tolerance is
   categorically unable to do this job, at any shape.

**Carry it forward from 2c on; do not retrofit the earlier kernels.** Converting kernels
that already pass is rework, and the tranche order is the priority. If the token-ID gate
later fails and implicates an early kernel, retrofit that one, with the failure as the
reason.

### A tolerance cannot detect a reordering, and no shape can fix that

The general statement, because it cost a wrong fix to learn and it is not about Vulkan.

`assert_close` compares at `1e-3·mx + 1e-3` — order `2e-3` on these shapes. Two different
summation orders over the same terms differ by order `1e-7`. **The tolerance is four
orders of magnitude larger than the perturbation it is being asked to detect**, so it
cannot see reordering at all.

The trap is that this looks like a shape problem. It was diagnosed as one: `mla_value`'s
oracle modelled a 32-lane strided accumulation, and at `kvl = 64` only 16 lanes had work
and each ran a single iteration, so the modelled order was provably unexercised. The
prescribed fix was to raise `kvl` to 256 so every lane loops. **That was done, and a naive
ascending sum still passed — at 27001× margin.**

> **The gap is between the SIZE OF THE PERTURBATION and the SIZE OF THE TOLERANCE, not
> between shapes.** Enlarging the shape changes which terms are summed, never the
> magnitude of a rounding difference relative to a fixed relative tolerance. No shape
> closes a four-order-of-magnitude gap.

This is the fourth failure class from `docs/measurement/probes/README.md` — *a check constitutionally
blind to the defect the code is exposed to* — wearing a new costume. The tolerance is not
broken and the oracle is not wrong; the check simply cannot see this class, and the code is
squarely exposed to it, because summation order is the entire reason `wave_sum` is a fixed
ladder rather than `subgroupAdd`.

The fix is a different KIND of assertion, not a better shape: compare bytes. With
bit-identity asserted, replacing the modelled ladder with `partials.iter().sum()` fails **10
of 15 elements**. The ladder is load-bearing for the first time.

> **Before tightening a tolerance-based test, ask whether the defect class you care about
> is even representable at that tolerance.** If the perturbation is orders of magnitude
> below the bound, every shape you try will pass and you will conclude the code is fine.

### The original design, for the record (superseded above)

The token-ID gate needs a whole forward pass. There is a cheaper version that works
*during* the port and localises a divergence to one kernel instead of to a token.

Both backends' oracles already run the **same input** — same `Lcg` seed formula, same
shapes — against the **same CPU reference**. So: hash the kernel's output buffer, commit
the HIP-produced hash as a golden, and have the Vulkan oracle assert against it. That
upgrades "the two backends report the same max error" from *consistent with* bit-identity
to *established*, per kernel, at the kernel — and it is simultaneously the golden-bits
regression test this repo has never had on either backend.

Evidence it is worth doing: `gemv_fp8` at 256x512 reports `err=9.537e-7`,
`margin=2928.2x` under **both** backends, identical to four significant figures. That is
suggestive and it is not proof, and the gap between those two is exactly what this
mechanism closes.

Build it once 2b and 2c have landed, so it covers eight or nine kernels rather than two.
Requirements when you do:

- Use the same FNV-1a as `tests/glsl_numerics.rs`, for the same reason: `DefaultHasher`
  is not stable across Rust releases, and a golden that breaks on a toolchain bump
  teaches people to regenerate goldens.
- **The failure message must say INVESTIGATE, DO NOT REGENERATE.** A golden refreshed on
  mismatch guarantees nothing, and refreshing is the path of least resistance at the
  moment it matters most.
- Record which backend produced each golden and on what hardware. A golden is a claim
  about a specific stack.

### Every accuracy-contract builtin in the remaining kernels, decided in advance

Enumerated by grepping all of `kernels/*.hip` for transcendentals BEFORE writing the
shaders, so the token-ID gate has a written list of suspects before it runs rather than a
mystery after it fails. Eliminate where possible; pre-register where not.

| Site | Builtin | Tranche | Decision |
|---|---|---|---|
| `common.hpp:45` e4m3 decode | `exp2f` | shipped | **ELIMINATED** — power of two built from bits, exact by construction |
| `linalg.hip:148` rmsnorm | `1.0f/sqrtf` | 2a | **CONSTRAINED** — must stay a divide, never `inversesqrt`. Rule 9 |
| `indexer.hip:48` | `1.0f/sqrtf` | later | same as rmsnorm; rule 9 already covers it |
| `linalg.hip:130` swiglu | `expf` | 2a | **PRE-REGISTERED** divergence — genuine transcendental |
| `moe.hip:26` silu | `expf` | 2d | **PRE-REGISTERED** — same mechanism as swiglu |
| `attn.hip:181-182` online softmax | `expf` | **2c** | **PRE-REGISTERED** — see below |
| `attn.hip:244` split combine | `expf` + `isfinite` | **2c** | **PRE-REGISTERED**, and `isfinite` must be composed |

**`exp` cannot be eliminated the way `exp2` was.** The e4m3 fix worked because the
argument was an integer and the result an exact power of two. Softmax and SiLU evaluate
`exp` at arbitrary runtime values, where no bit-exact construction exists — and there are
THREE implementations in play, not two: Rust's `f32::exp` in the oracle, HIP's `expf`, and
GLSL's `exp`. These kernels are tolerance-tested on both backends and always have been.
Pre-registration is the honest resolution, not a concession.

### DECIDED, before writing 2c: `exp` is pre-registered, and the oracle SPLITS

The decision, with its mechanism, taken as a design step rather than after a red oracle.

**`exp` cannot be eliminated, and the reason is structural rather than effortful.** The
`exp2` elimination worked because its argument was an exact integer and its result an exact
power of two, so it could be built from bits. Softmax evaluates `exp` at arbitrary runtime
values. There is no bit-exact construction, and there are THREE implementations in play —
Rust's `f32::exp` in the oracle, HIP's `expf`, GLSL's `exp` (specified to 3 ULP). Pre-
registration is the honest resolution.

**The consequence the FMA finding forces, and it is the important one: 2c CANNOT have a
bit-exact oracle the way 2b does.** Tranche 2b's oracles reach `err = 0.000e0` because
every operation in those kernels is `+`, `*` or a table lookup, all reproducible exactly in
Rust once contraction is modelled. Rust's `exp` and GLSL's `exp` are different functions,
so any output downstream of an `exp` is unreachable by an exact CPU reference — no amount
of care fixes that.

And a tolerance is **categorically blind to reordering** (see above). So a naive 2c oracle
would be blind to exactly the class `mla_latent_attend` is most exposed to: it carries a
`SUBW`-strided reduction, a shuffle ladder, and a per-tile accumulation order.

**Therefore the oracle splits along the `exp` boundary:**

| part | arithmetic | oracle |
|---|---|---|
| score `s = scale·(qa·L_tt + qr·R_tt)` | `+`, `*`, shuffle ladder | **BIT-EXACT** — this is where the ordering risk lives, and it is `exp`-free |
| tile widening `e4m3f(Lc8)·Lscale`, `bf16f(Rc)` | table + `*` | **BIT-EXACT** |
| online-softmax update, split combine | `exp` | tolerance, **pre-registered** |

The score reduction is the part with the ordering hazard and it sits entirely *before* the
first `exp`, so it can be tested at full strength. Do that rather than accepting a single
end-to-end tolerance that hides it.

**A SECOND, INDEPENDENT BIT-IDENTITY HAZARD IN 2c, VISIBLE ONLY BECAUSE OF THE FMA
FINDING.** The accumulator update is

```c
acc[k] = acc[k] * corr + p * Lt[tt * kvl + lane + k * SUBW];   // attn.hip:185
```

**Two multiplies, one add — so there are two valid contractions**, `fma(acc, corr, p*Lt)`
and `fma(p, Lt, acc*corr)`, and they give different results. Nothing in either language
specifies which a compiler picks. Every previously ported kernel had at most one multiply
feeding an add, so the contraction was unambiguous and the two backends agreed by luck of
having no choice to make. Here they can genuinely diverge, and it has nothing to do with
`exp`.

The same shape appears at `l = l * corr + p` (unambiguous — one multiply) and at
`sum += partial[..] * w[s]` in the combine (unambiguous). It is `acc[k]` specifically.

**MEASURED, BEFORE WRITING THE SHADER — HIP's choice is now a known constraint rather than
a suspect.** `hipcc --offload-arch=gfx1151 -O3 --cuda-device-only -S kernels/attn.hip`, in
`mla_latent_attend`'s unrolled accumulator loop:

```
v_mul_f32_e32  v42, v28, v42    ; v42 = acc[k] * corr   — PLAIN multiply
v_fmac_f32_e32 v42, v1,  v5     ; v42 = fma(p, Lt[k], v42)
```

repeated once per unrolled `k`. So HIP evaluates

```
acc[k] = fma(p, Lt[k], acc[k] * corr)
```

— it fuses the `p·Lt` multiply and leaves `acc·corr` unfused. **That is the form the GLSL
must produce**, and it is now checkable rather than hopeful: write the shader, disassemble
it, and confirm RADV fuses the same side. If it picks the other, force the grouping
explicitly on both sides — the cheapest lever is a named temporary for `acc[k] * corr`,
which removes the compiler's freedom to choose.

Note how cheap this was. The GPU is the scarce resource and the compiler is not: this
question was answered on the CPU in seconds, before a line of 2c existed, exactly as
`docs/measurement/benchmarks.md`'s "Read the ISA before you book the device" prescribes. It also would not
have been ASKED without the FMA finding, which is the argument for recording mechanisms
rather than symptoms.

**2c also carries three same-spelling hazards at once.** Meeting them as design decisions
beats meeting them as a red oracle:

- `expf(m - m_new)` and `expf(s - m_new)` in the online-softmax update, both with
  arguments `<= 0` by construction, so results land in `(0, 1]`. Softmax is a ratio, so
  errors partially cancel between numerator and denominator — partially, not exactly.
- `isfinite(m_g)` at the split combine. **GLSL has no `isfinite`**; it must be composed
  as `!isinf(x) && !isnan(x)`, and getting the polarity backwards zeroes every split
  weight instead of guarding the empty-split case.
- `__shfl(part, 0, SUBW)` — a broadcast whose `width` parameter partitions the wave.
  **SPIR-V subgroup ops have no width parameter at all.** This one is currently safe
  only because `SUBW == WAVE == 32`, so the partition is the whole wave — a coincidence
  of a constant, not a property of the code. If `SUBW` ever differs from `WAVE`, a
  transliteration breaks silently. Port it as an LDS broadcast, consistent with `wave_sum`
  already being an LDS tree here.

**PRE-REGISTERED: what happens if `swiglu` breaks this gate.** `swiglu` computes
`x/(1+exp(-x))`, and `exp` is a library function whose last bits are unspecified in both
GLSL and HIP — two correct implementations disagree. It is therefore the first suspect if
the token-ID comparison fails, and the options are written down **now, before the
result**, because choosing after a red gate is when motivated reasoning is strongest:

  (a) **Implement a bit-exact `exp` on both sides.** Expensive, and it changes the HIP
      kernel — so it needs its own oracle and its own quality gate before it can be
      trusted. Buys a genuinely identical backend.
  (b) **Redefine the gate as identical token IDs for K tokens, K MEASURED not chosen.**
      Honest and weaker. Makes the divergence point a reported number rather than a
      hidden one, and a K in the hundreds is still strong evidence.
  (c) **Accept and document the divergence, with the mechanism named.** Cheapest, and
      only defensible if the divergence is shown to be confined to `exp` — which means
      demonstrating it, not asserting it.

The gate may well pass regardless: a low-bit `exp` difference has to survive 78 layers
and a 154,880-way argmax to flip a single token. If it does pass, this paragraph cost
nothing and is the contingency that was not needed.

**Expect to fight the toolchain for it.** This is not a bar you meet by writing careful
GLSL. `glslc -O` rewrote `a / 448.0` into a multiply by `fl(1/448)` under an explicit
bit-identity contract, and 55.1% of block scales diverged — see "The toolchain rewrites
float arithmetic" above. Meeting cross-backend identity will need more defences of that
shape: operands the optimiser cannot see, more build-time rejections, and disassembly of
anything whose exactness matters. Whoever attempts this should know that before the
first mismatch, not after a day of auditing their own shader.

Until then, do not describe the port as verified end to end. The oracles are per-kernel
evidence, and **nothing in this repo tests composition on either backend** — which is
why that byte-identity result arrived as a side effect of a performance experiment
rather than from a test.

Scoped, not "unmodified", on purpose: the suite also covers `gemv_vq`/`gemv_i4`, which
are microbench kernels the decode path never calls. Gate those oracles on the ported
set (`#[cfg]` or a skip list) rather than porting two kernels to satisfy a test.

### Two rules for writing the oracles

Both are enforced in `tests/vk.rs`'s header, where the next author will be; repeated
here because this is where the tranche gets *planned*.

**Write the oracle without reading the shader.** Derive it from the HIP original and
`src/math.rs`, which are the specification. An oracle written by someone who has seen
the implementation is a consistency check wearing a correctness check's clothes — it
agrees because it was copied, and it will ratify a shared misreading of the HIP. The
`fwd.hip` oracles were written under that constraint deliberately, and the rework when
the two disagree *is* the signal.

**A byte-exact oracle must prove its inputs are unambiguous.** Where a test compares
quantised bytes, the test DATA is a source of cross-driver flake independent of the
code: a value on a rounding midpoint of the target format can legitimately quantise
either way, so the driver's arithmetic accuracy decides the comparison rather than the
shader. Green here, red elsewhere, shader innocent — and the person debugging starts in
the kernel, because that is where the failure appears.

`tests/vk.rs::assert_quantization_unambiguous` is the check. Give it the margin **the
spec guarantees**, not what this GPU delivers: Vulkan promises 2.5 ULP on `FDiv`, and
`FAdd`/`FMul` are correctly rounded but a reduction accumulates, so a dot product needs
a margin that grows with its length. This bites hardest in the **fp8 MoE tranche**,
which compares quantised bytes throughout and where a seed-dependent flake is most
expensive to diagnose. Make the failure name the seed, not the shader.

Do not accept "close enough" numerics. The oracle tolerances were tuned against real
failures — bf16 codebooks failed the 1e-3 oracle and fp16 passed. That headroom is the
safety net for a second backend, so it is worth knowing how much of it there actually
is.

### Measured oracle headroom (HIP, `--features rocm`)

The previously recorded "~5.6× margin" for the fp16 codebook **was not a real margin.**
It was measured under `tests/kernel.rs`'s `Lcg`, which shifted a `u64` right by 33 and
so returned values in [-1, -2.3e-10] — every sample negative. With both operands
negative every product in a matvec is positive, so the partial sums grow instead of
cancelling, the max-magnitude term in `1e-3 * mx + 1e-3` inflates, and the threshold
inflates with it. Fixed in `ceb759a`; these are the numbers with balanced inputs:

| oracle | err | tol | margin |
|---|---|---|---|
| `moe_vq` | 4.841e-2 | 1.612e-1 | **3.3×** |
| `gemv_vq` | 4.036e-3 | 2.818e-2 | **7.0×** |
| `gemv_fp8_splitk` | 1.669e-5 | 9.781e-3 | 586× |
| `moe_i4_real` | 9.835e-7 | 1.758e-3 | 1787× |
| `moe_i4` | 9.918e-5 | 2.391e-1 | 2411× |
| `gemv_fp8` | 9.537e-7 | 2.793e-3 | 2928× |
| `mla_absorb` | 2.384e-7 | 1.854e-3 | 7776× |
| `mla_value` | 4.768e-7 | 2.108e-3 | 4421× |
| `mla_attend` | 8.382e-9 | 1.008e-3 | 120203× |
| `vadd` | 0 | 3.000e-3 | exact |

**3.3× is oracle headroom, not output quality.** It says how much room a reimplementation
has before the *test* fails; it says nothing about whether the model's outputs are 3.3×
away from wrong. Do not quote it as a quality figure.

**The consequence for this port.** The two tightest oracles are the VQ paths, and the VQ
MoE kernels — `moe_gateup_vq`, `moe_down_vq` — are exactly the ones not yet ported and
the ones the plan already calls the hardest. A reduction-order difference there has 3.3×
to fit inside, not the 5.6× the plan used to claim. So: **if a VQ oracle fails during the
port, the first hypothesis is a genuine ordering or gather bug, not a tolerance that
needs loosening.** Widening a bound on the kernel with the least headroom in the suite
would discard the only signal that would have caught the bug.

(This table belongs in `docs/measurement/benchmarks.md` at merge time — it is kept here for now to avoid
a conflict with concurrent edits to that file.)

## Staging

Each phase ends with something runnable; no phase leaves the tree broken.

1. **Spike** — instance/device/queue init, feature detection, one kernel (`gemv_f32`)
   dispatched from a GLSL shader, its oracle test green under `--features vulkan`.
   Proves the waist and the build plumbing. Small; do it before committing to the rest.
2. **Memory + sync** — `DeviceBuf`/`DeviceTier`/`VmmBuf` equivalents, timeline-semaphore
   `Signal`, and the device-local copy pool for the io_uring bounce path. No profiling
   yet. Exit: `DeviceTier` reserve/copy tests pass; the reaper's H2D path round-trips.

   **DELIVERED, SMALLER THAN THIS.** `DeviceBuf`/`DeviceTier`/`VmmBuf`, `memcpy_dtod`,
   and the timeline-semaphore `Signal` are done and tested. Two parts of the criterion
   are NOT met and are deferred rather than reinterpreted:
   - the **copy pool** is deliberately not built — see the deviation above;
   - **"the reaper's H2D path round-trips"** cannot be met at all while the reaper is
     `rocm`-only (`stream.rs`, `asyncfetch.rs`), so it moves to the io_uring port in
     phase 4, where there is something to round-trip.

   Also unfinished, and flagged because it looks done from the outside: `Gpu::signal`
   covers work already SUBMITTED, not work merely recorded. Arming on top of recorded
   work needs the single command buffer to become a small ring. Deferred to phase 4 for
   the same reason as the pool — no consumer exists yet to hold the design honest.
3. **Kernels, oracle-first** — port in test order: `fwd` → `linalg` (non-MoE) → `mla` →
   `attn` → `moe`. Exit: the 16 ported kernels' oracles green.
4. **Integration** — `backend.rs` switch, `pin.rs`/`gpu.rs` imports, end-to-end decode
   on the 4-layer stub, then the full artifact. Exit: coherent output.

   **READ "Phase 4 needs two queues, not one" BELOW FIRST.** The exit criterion above is
   insufficient on its own, and the seam is smaller than it looks.

   **INCREMENT 1 DELIVERED, and it decodes.** See "Increment 1: measured" below for what
   landed, what the numbers were, and the two findings the run produced.

   **INCREMENT 2 DELIVERED, and it implements the design.** Three queues, the ring, the async
   staging copy, and the timestamp query pools this table filed under phase 5. See
   "Increment 2: measured". **Phase 4 is complete.**
5. **Bench + profiling** — ~~timestamp query pools~~ **DONE, and pulled into phase 4**: an
   overlap invariant that cannot be measured is assumed rather than upheld, so the instrument
   had to ship with the thing it certifies. `Stamp` in `vk.rs` feeds `backend::Event`, which
   `telemetry.rs` already consumed. What remains of this item is the comparison work: the
   port IS slower, the per-phase spans now say WHERE (MoE kernels, 2.1x), and the rule still
   holds — record, do not tune during the port.

**Sequence this last.** Of the three open proposals it has the least measured upside:
PILOT and `top-m` attack a bottleneck we have quantified, while this buys portability
we do not currently need on a node where ROCm works. Build it when the portability goal
is real, not because the plan exists.

> **2026-08-01: the sequencing argument inverted, and the losers are both gone.** `top-m`
> was RETIRED 2026-07-30 (+3.63% PPL on int3-vq, +12.7% on int4 against a ~1% bar) and
> PILOT was DELETED 2026-07-31 (the eviction veto bound on 0.9% of evictions). **This item —
> the one ranked last of three — is the one that shipped.** Kept because the ranking was
> reasonable on the evidence available and still lost: "least measured upside" priced the
> two competitors' *modelled* upside against this one's *certain* cost, and modelled upside
> is the term that failed. See `cache-conditional-routing.md` and `cross-layer-prefetch.md`.

## 2d pre-flight: `moe.hip`, the three int3-vq kernels

Scoped against the source before writing, so the tranche starts from decisions rather
than discoveries. **Written and merged** (`047d0be`, `bffdc9d`); the `moe_*_vq` and
`moe_reduce` shaders exist and the header's "landed through tranche 2d" is the current
status. This section is retained as the plan it was executed from.

> **2026-08-01: `moe_reduce.comp` no longer exists**, nor does the HIP kernel, its C
> wrapper, either launcher or the Vulkan push struct. Fixed-point accumulation
> (`reference/architecture.md` §12) replaced it on 2026-07-31 — `moe_acc_drain` is the
> shader that ships. Decision 4 below, "`moe_reduce` is a partition feeding an
> order-sensitive reduction, so it is a numerics decision, not scheduling", was **correct
> and is exactly why it went**: the replacement makes every expert `atomicAdd` at a fixed
> integer scale, so the partition carries no freedom and the schedule cannot reach the
> result. The plan is retained unedited; `moe_*_vq` did land as written.

**The tightest oracle in the suite guards this tranche.** `moe_vq` has **3.3×** headroom
against `gemv_fp8`'s 2928×. The standing rule applies without restatement: *a failing VQ
oracle is a gather or ordering bug until proven otherwise, never a tolerance to loosen.*
Widening the bound on the kernel with the least headroom would discard the only signal
that would have caught the defect.

**Five porting decisions, each already determined:**

1. **The fp16 codebook needs no 16-bit storage extension.** `dot_vq_wave` reads two
   `__half2`; GLSL's `unpackHalf2x16(uint)` returns exactly that pair from a `uint` word,
   and half→float is exact. So the codebook is read as `uint` words like every other
   packed tensor here, and `VK_KHR_16bit_storage` stays unrequired — one fewer device
   precondition, which after the BALLOT finding is worth having.
2. **`ExpertDescVq` is six device addresses per expert**, so `descs[e]` is a `uint64`
   buffer read at stride 6. This is the tranche that actually exercises
   `buffer_device_address` as the plan intended, and the push-constant budget prediction
   (~88 bytes with the descs pointed-to rather than inlined) should be MEASURED here
   rather than assumed — `push_struct!` will fail the build if it is wrong, which is the
   right failure.
3. **The 12-bit index unpack straddles word boundaries.** HIP reads
   `idxrow[byte] | (idxrow[byte+1] << 8)` — two adjacent bytes, which with word-only
   loads may span two words. The `WORD` padding already covers the tail read; the
   straddle needs explicit handling, and it is the single most likely place for a silent
   gather bug in this tranche.
4. **`moe_reduce` is a partition feeding an order-sensitive reduction**, so by the rule
   established in this document it is a numerics decision, not scheduling. Its `e_count`
   batching must be transliterated, not re-derived, for the same reason `mla_plan_splits`
   was.
5. **`siluf` carries the pre-registered `exp` divergence**, confirmed present in the ISA
   (`v_mul_f32 0xbfb8aa3b` — the negative log2(e), since it evaluates `exp(-x)`). Same
   status as `swiglu`; nothing new to decide.

**MEASURED, and it is the hazard this document predicted would recur.** `moe_gateup_vq`'s
inner subvector dot is `x.x*c0 + x.y*c1 + x.z*c2 + x.w*c3` followed by
`acc += scale * dot` — four multiplies feeding three adds, then another multiply-add. That
is the two-multiplies-one-add shape from `attn.hip:185`, but with more freedom. hipcc's
instruction mix for the kernel:

    6 x v_mul_f32   4 x v_fma_f32   3 x v_fmac_f32   1 x v_dual_fmac_f32

So contraction choices **are** being made here and the port must match them rather than
hope. The first task of 2d is the isolate-and-disassemble treatment that
`attn.hip:185` got — a minimal kernel calling `dot_vq_wave` alone, read term by term —
because the mix above is not attributable to specific products while inlined into the
whole kernel. Do that before writing the shader, not after the oracle disagrees.

### Three places the two backends legitimately differ, so nobody reads them as porting bugs

Surfaced while writing 2d's oracles. All three are recorded rather than fixed — the first
is a pre-existing HIP-side issue that costs nothing today, and moving `main` would diverge
the bases of concurrent work for no benefit.

**1. `math.rs::silu` is not bit-identical to `moe.hip::siluf`, and the existing ROCm test
cannot see it.** `kernels/moe.hip:25` computes `x / (1.0f + expf(-x))` — a DIVIDE.
`src/math.rs` computes `x * sigmoid(x)`, a reciprocal-then-multiply. Those round
differently. `tests/kernel.rs::moe_vq_matches_reference` compares the kernel against
`math::silu` at `1e-3 * mx + 1e-3`, which is orders of magnitude looser than the
disagreement, so it passes and always has.

> **The concrete consequence: `moe_vq_matches_reference` CANNOT be tightened past ~1e-3
> until its reference is changed to divide.** Anyone who tries — and tightening a tolerance
> is exactly what a later bit-exactness push would do — will find a mismatch that looks
> like a kernel defect and is in the oracle. That afternoon is what this paragraph exists
> to save.

The Vulkan shader writes the divide, matching the HIP. So the two BACKENDS agree; it is
`math.rs` that is the outlier, and the standing rule (mirror HIP, not `math.rs`, because
the backends are what must agree) resolves it the same way it resolved the bf16 NaN case.

**2. The Vulkan launcher is deliberately STRICTER on VQ dimensions.** `moe.hip` never
checks `hidden % VQ_GROUP` or `inter % VQ_GROUP`; its `vq_rb`/`vq_ng` are raw integer
divides that truncate silently, so a bad dimension mis-sizes every row on device with no
diagnostic. `launch_moe_expert_range` rejects it. That is one of the few places this
backend refuses input the HIP accepts, and it is deliberate: the Rust `vq_row_bytes` /
`vq_groups` already `debug_assert` the same condition, so the guard makes the release-build
device path agree with the debug-build host path.

**3. The 16-byte alignment requirement on `x` is a HIP-only artefact, and the port
correctly does not inherit it.** `dot_vq_wave` in `common.hpp` reads
`*(const float4*)(x + t*VQ_DIM)`, which needs `x` 16-byte aligned. The GLSL twin reads four
scalar floats through a `buffer_reference`, so it has no such constraint — only the 4-byte
alignment every f32 access already implies. A reader comparing the two launchers will find
an alignment guard on one side and none on the other; that asymmetry is correct.

## THE QUALITY QUESTION K=0 LEFT OPEN: ANSWERED, AND VULKAN PASSES

Gate A returning K=0 (below) means nothing in A + C + D speaks to Vulkan's output
quality. `bin/ppl` was the instrument this document reserved for exactly that case. Run:

| | ROCm | Vulkan |
|---|---:|---:|
| PPL (762 teacher-forced tokens) | 5.275434 | **5.231609** |
| expert hit % | 78.08 | 78.05 |

Paired per-position, ROCm as baseline: **mean dNLL −0.00834, sd 0.1689, SE 0.00612,
95% CI [−0.02033, +0.00365], worse% 48.6.**

**Read it as NO DETECTABLE DIFFERENCE, not as an improvement.** `worse%` of 48.6 says
Vulkan is worse on about half the tokens and better on the other half — float noise with
no systematic direction — and the interval spans zero. `ppl.rs` itself flags a
better-than-baseline result as "implausible, suspect a bug"; the right conclusion is that
the two backends are indistinguishable here, not that Vulkan improves the model.

**And it is a POWERED pass, which is the distinction that matters.** The 1% bar is 0.00995
nats and the CI upper bound is +0.00365, so the interval EXCLUDES harm greater than 1%.
This is not the underpowered null `ppl.rs` warns against — it genuinely bounds the harm.

Why this works when K=0 does not: `--ppl` is TEACHER-FORCED. Both backends score the same
fixed token sequence, so it does not require them to agree on any argmax. That is precisely
why it survives the divergence that makes gate A vacuous.

**Scope, so this is not over-quoted.** 762 tokens, one corpus, `int3-vq / dense / lru` —
the only configuration both backends can run. It measures logit quality, not generation
dynamics, and it says nothing about the deferred int4/hybrid or DSA paths. The ROCm arm
reproduced the published int3-vq baseline (5.275434) exactly, so the reference is sound.

**Verdict: the Vulkan backend is as good, within 1%, on the configuration it supports.**
That is the question the gate was built to answer and could not.

## GATE A: K = 0. RECORDED HERE, THE FIRST TIME IT WAS MEASURED

The gate requires K "measured and REPORTED", with the baseline written down the first
time, because "a later collapse from hundreds to single digits is only visible as a
regression if the earlier value is written down". This is that record.

**K = 0.** Matched runs — int3-vq / dense / lru, `--max-mem 115`, 512 greedy tokens, same
artifact and prompt, ids dumped with `--dump-ids`:

```
# rivoli-ids v1 backend=rocm   mode=int3-vq policy=lru attn=dense tokens=512
# rivoli-ids v1 backend=vulkan mode=int3-vq policy=lru attn=dense tokens=512
first divergence at index 0:  rocm=2  vulkan=48148
overall agreement 5/512 = 1.0%
```

The backends disagree on the **first token** and never resynchronise. An earlier "identical
token IDs" observation was at `-bench 2` on a different prompt and is not a counterexample
worth keeping: two tokens is inside the noise of a coin flip.

**This is consistent with what this document already measured**, and is not a new defect:
`swiglu` differs in **1463 of 4096** outputs between the backends, and that feeds every MoE
down-projection, so by layer 78 the logits differ enough to move the argmax. The doc said
byte-identical output "CAN NEVER PASS"; K=0 is that statement's quantitative form.

**But it changes what the accepted gate is worth, and that has to be said plainly.** The
reasoning for dropping B rested on K carrying B's weight — "it is the only part of the gate
with any sensitivity to numerical drift at all". At K=0 it carries none. The blind spot the
doc accepted as *bounded* is in fact **total**: nothing in A + C + D speaks to Vulkan's
output quality.

Both backends produce coherent, on-topic, non-degenerate text (lrb 6, distinct 0.538 vs
0.555) — they simply produce *different* text. So this is not evidence the Vulkan backend is
wrong. It is evidence that the gate cannot tell us whether it is.

**Consequence, and it is the obligation this document already wrote down:** `bin/ppl` is now
the ONLY instrument with any purchase on Vulkan output quality. Running it is no longer a
nice-to-have that "stops being a merge condition" — it is the only remaining way to answer
the question the gate was built to answer. Do that before trusting a Vulkan decode for
anything but throughput work.

## INCREMENT 1 SHIPPED THE FAILURE THIS SECTION WARNS ABOUT

> **RESOLVED by increment 2** — three queues, the command-buffer ring, an async staging
> copy, and real timestamp query pools. Measured: **97% of fetch hidden**, up from 0%.
> See "Increment 2: measured" below. This section is retained as the record of the process
> failure, and its list of three violations is the checklist increment 2 was built against.
> It also undercounted by one, which is the most useful thing about it now: see item 4
> there.

Recorded because the warning was read, quoted, and then overridden — which makes it a
process failure, not an oversight.

Increment 1 (merged) wired the backend waist and got the engine decoding under
`--features vulkan` on a SINGLE queue, deferring the queues to a later increment. The
section below is titled "Phase 4 needs two queues, not one". It was followed to the
letter for the kernels and abandoned for the integration.

**Measured, matched (int3-vq / dense / lru, `--max-mem 115`, 512 tok, same prompt):**

| | ROCm | Vulkan increment 1 |
|---|---:|---:|
| tok/s | 2.52 | 1.44 |
| moe/tok | 276 ms | **522 ms** |
| route/tok | 104.6 ms | 128.2 ms |
| expert hit | 75.6% | 76.5% |

`moe` +89% against `route` +23% — the regression sits precisely in the phase whose design
IS the overlap. ~246 ms/token of it.

**Three violations, not one.** The queues were the visible one:

1. one queue, so fetch cannot overlap compute;
2. `stream.rs`'s `stage::copy_to_slot` under `vulkan` was a synchronous
   `std::ptr::copy_nonoverlapping` — the host CPU copying 2.16 GB/token instead of a DMA
   engine, blocking the reaper thread;
3. `Event::elapsed_ms` returned 0.0, so `compute_gpu_ms` was 0 and `fetch_hidden_pct`
   reported **0% as an arithmetic artifact of the stub**, not a measurement.

(3) is why this should not have shipped. **An invariant that cannot be measured is not
upheld, it is assumed** — and the reported "0% hidden" would have been quoted as a
finding by anyone reading the run. Timestamp query pools were filed under Phase 5; they
belong to Phase 4, because they are the instrument that certifies the thing Phase 4 exists
to preserve.

The lesson generalises past Vulkan: this document already says a green correctness check
"means less than it appears to" here. The fix is that **the throughput criterion is part of
the gate, not a follow-up** — see the acceptance criteria below, which increment 1 did not
have to satisfy and should have.

## Phase 4 needs two queues, not one — and the acceptance gate cannot see it

Written before integration starts, because the failure it describes is invisible to every
check this port has: **the token-ID gate passes a backend that is three times slower.**

### The streaming layer is not ported, and has not strayed only because it does not exist

> **SUPERSEDED BY INCREMENT 1 — see "Increment 1: measured" below.** The two subsections
> here diagnose the state of the tree BEFORE integration, and both have since been acted
> on: the four modules named are now `any(rocm, vulkan)` and `crate::backend` has three
> consumers. They are kept verbatim because the second one is the reason the increment was
> scoped the way it was, and because the concurrency argument that follows them is still
> the plan for increment 2.

Verified rather than assumed: `gpu.rs`, `pin.rs`, `asyncfetch.rs`, `gpustream.rs` and
`stream.rs` all carry `#![cfg(feature = "rocm")]`, so a Vulkan build compiles none of
them. That is correct for phases 1–3 and is the whole reason the architecture has not
drifted. It also means every claim below is about work not yet begun.

### The seam is smaller than `backend.rs` claims, in two ways

1. **`crate::backend` has no consumers at all.** `gpu.rs` imports `crate::hip::{...}`;
   `pin.rs` imports `crate::hip::memcpy_dtod`. A grep for `crate::backend` returns only
   that file's own doc comment. The waist is built and nothing goes through it.
2. **The `launch_*` surface is not the boundary.** `gpu.rs` and `asyncfetch.rs` import
   `crate::gpustream::{HipStream, HipEvent, Signal, stream_signal}` DIRECTLY, and
   `HipStream` has no Vulkan analogue. Re-exporting `vk::*` from `backend.rs` cannot
   supply a stream abstraction that does not exist.

Both corrected in `backend.rs`'s own comment, which previously promised that swapping
backends was "a cargo flag and nothing else."

Increment 1's answer to (2): the stream types are re-exported from `backend.rs` under
BACKEND-NEUTRAL names (`Stream`, `Event`), and `src/vkstream.rs` supplies the Vulkan side
— a `Stream` that carried nothing because the launchers ignored their `_stream` argument, and
an `Event` whose `elapsed_ms` was always `0.0`. Neither pretended to be what HIP has; both
were documented at the type as the absence they were.

**Increment 2 made both of them real.** `Stream` is now the NAME of one of three queues
(`Stream::compute()` / `Stream::fetch()`, so the role is chosen by the consumer that knows
it), the MoE launchers parse and honour it, and `Event` is a `VK_QUERY_TYPE_TIMESTAMP` query
pool. The one part of the shape that survived unchanged is the right one: `*mut c_void`
remains the token, because the signature is shared with HIP — but it is now a TAG that
`Q::parse` refuses if unrecognised, never a pointer anything dereferences.

### The performance property lives in the CONCURRENCY, and vk.rs has none

The engine's headline behaviour — fetch overlapping compute on **two independent streams**
— is what this section is about. (The "~95% hidden" number it originally cited was
retracted 2026-07-30: it came from a bracket containing its own stalls, and the engine is
in fact fetch-bound. See ARCHITECTURE.md §3. The *structural* argument below is unaffected
— concurrency is still exactly what makes the overlap possible, and matters more now, not
less, since fetch is the binding constraint.)

| piece | where |
|---|---|
| the reaper's dedicated fetch stream | `asyncfetch.rs`, `HipStream::new()` in `AsyncFetch::new` |
| the MoE expert compute stream | `gpu.rs`, `compute_stream`, explicitly "separate from the null stream the rest of the forward uses" |
| the driver that races them | `gpu.rs`, `try_for_each_concurrent` over the expert descriptors |

`vk.rs` today is **one queue behind one `Mutex<Cmd>`**, documented as mirroring "HIP's
default stream". Integrating onto that serialises fetch against compute.

> **HISTORICAL from here to the end of this section.** `vk.rs` now has three
> `Mutex<Stream>`, one queue each, and the overlap is measured at 97% — see "Increment 2:
> measured". The analysis below is kept because it is the reasoning the fix was built from,
> and because its final subsection is the record of the increment that ignored it.

**And the acceptance gate would not notice.** Token IDs depend on arithmetic, not on
overlap; a fully serialised backend computes exactly the same numbers. So the gate this
document spent pages sharpening — identical token IDs, greedy decode, pinned conditions —
passes a build whose central performance property has collapsed. **Phase 4 therefore needs
a throughput criterion alongside it: measure fetch-hidden % and tok/s, and compare against
the ROCm numbers in `docs/measurement/benchmarks.md` at matched `--max-mem` and `--cache-policy`.** A
correctness gate cannot certify a concurrency structure, and this is the clearest case in
the project of a green check meaning less than it appears to.

### The hardware permits the honest fix, and it merges with a deferred item

Measured on this device (`vulkaninfo`, families with their queue counts):

| family | queues | flags |
|---|---:|---|
| 0 | 1 | GRAPHICS \| COMPUTE \| TRANSFER \| SPARSE |
| 1 | **4** | **COMPUTE \| TRANSFER \| SPARSE** (compute-only) |
| 2–4 | 1 each | video decode / video encode / sparse binding |

`Gpu::create_device` already selects the compute-only family — `compute(f) && !GRAPHICS`,
which is family 1 — and requests exactly one queue from it (`prio = [1.0f32]`). **So the
second queue needs no new family-selection logic: ask for two priorities instead of one.**
HIP's two streams map onto two queues from the family already chosen, ordered by the
timeline semaphore that is already built and measured (19.28 µs/signal).

This also supplies the forcing function for the item deferred in phase 2: `Gpu::signal`
covers work already SUBMITTED, not work merely recorded, because there is a single command
buffer. A second queue needs its own recording state anyway, so **the command-buffer ring
and the second queue are one design, not two** — which is the argument for doing them
together rather than bolting the ring on later.

### The design, on paper

Written before any integration code, and structured so the command-buffer ring falls out
rather than being bolted on.

> **BUILT, in `vk.rs`, and it held up.** Three deviations, each measured or reasoned rather
> than preferred, plus one thing this text got wrong:
>
> - **`RING = 16`, not 4.** 4 stalls the DECODE thread — the one driving the expert stream —
>   because the MoE stream flushes once per expert and a layer submits ~9 buffers. The doc is
>   right that an undersized ring is a stall not a bug; it is wrong that the stall is
>   harmless, because this particular thread stalling is the overlap.
> - **`Stream::next` is a plain `u64`, not an `AtomicU64`.** It is only ever touched under
>   the stream's own mutex, and an atomic would imply — falsely — that allocating a value
>   outside the lock is safe. It is not; that is the monotonicity argument for per-stream
>   timelines in the first place.
> - **A device with fewer than three queues ALIASES** several `Q`s onto one `Stream`, rather
>   than handing one `VkQueue` to two mutexes. The doc assumes family 1's four queues, which
>   this device has; the aliasing is what keeps the portability claim honest, and it warns.
> - **The signal is registered AFTER the submit, not before.** The waiter re-reads the
>   counter every pass and waits on values that may already be reached, so there is no
>   missed-wakeup window — which deletes the unregister-on-failure path the single-queue
>   version needed.
> - **What this text missed: RECORDING IS NOT SUBMITTING.** See item 4 of "Increment 2:
>   measured". Three queues with lazy recording still serialise fetch against compute.

**THREE streams, not two.** The earlier text said two and that undercounts. HIP runs:

| stream | owner | work |
|---|---|---|
| the null/default stream | `gpu.rs`, implicitly | the whole forward pass except MoE experts |
| `compute_stream` | `gpu.rs` | MoE expert partials, explicitly "separate from the null stream" |
| the fetch stream | `asyncfetch.rs`, in the reaper | H2D copies of streamed expert weights |

The overlap that hides 95% of fetch is between the last two; the first exists because the
rest of the forward must not be reordered against either.

**Three queues from family 1, which needs no new selection logic.** `create_device`
already picks the compute-only family (`compute(f) && !GRAPHICS`) and asks for one queue
(`prio = [1.0f32]`). It becomes `[1.0; 3]`, and family 1 has four. **Every queue is in the
SAME family, which deletes a whole category of work:** no queue-family ownership transfers
on any buffer, ever. That is the single biggest simplification available and it is why
using the graphics family for anything would be a mistake.

**One `Stream` per queue, replacing the single `Cmd`.**

```
struct Stream {
    queue: vk::Queue,            // stays INSIDE the mutex, as Cmd's does today
    pool: vk::CommandPool,
    ring: [vk::CommandBuffer; RING],
    done: [vk::Fence; RING],     // retires slot i
    head: usize,
    recording: bool,
    timeline: vk::Semaphore,     // ONE PER STREAM, see below
    next: AtomicU64,
    poisoned: bool,
}
```

`Gpu` holds three `Mutex<Stream>`. `vkQueueSubmit` requires external synchronisation
per-queue, and a mutex per stream gives exactly that — the same borrow-checker enforcement
the current design gets by keeping `queue` inside the guard, now three times over. It also
means the fetch stream never blocks on the compute lock, which is the entire point.

**A TIMELINE PER STREAM, not one shared.** A timeline semaphore must be signalled with
strictly increasing values, so a shared one would force a global order across queues —
reintroducing exactly the serialisation being removed, and making value allocation a
cross-queue critical section. Per-stream timelines make each queue's values trivially
monotonic under its own lock. Cross-queue ordering is then expressed the natural way: a
submit on stream A waits on stream B's timeline at a value B has already been told to
signal. The waiter thread handles several with one `vkWaitSemaphores` and `WAIT_ANY`.

**The ring is what closes the `Gpu::signal` gap, and that is why they are one design.**
Today `signal()` covers work already SUBMITTED, because there is a single command buffer:
arming on merely-recorded work would need to submit it, and the buffer cannot be reused
while pending. With a ring, `flush()` is *end the current buffer, submit it with a
timeline signal, advance `head` to a slot whose fence has retired* — so arming on recorded
work is just `flush()` then hand back the value. No separate mechanism. `RING` is sized by
how many flushes can be in flight before the oldest must have retired; 4 is a starting
guess and the fence wait makes an undersized ring a stall rather than a bug.

**`device_sync` joins all three**, in a fixed order (main, MoE, fetch). It is once per
token, so the cost is three fence waits rather than one, and the ordering is fixed so the
join itself cannot become a source of nondeterminism.

### Increment 1: measured. It decodes, it agrees with HIP, and it is serialised.

`--features vulkan --bin rivoli` builds and decodes. **Matched runs**, same artifact
(`/var/db/rivoli/glm52-vq3-full`), same flags (`--mode int3-vq --attn dense --max-mem 60
--cache-policy 2q`, `-bench 2`), same prompt, greedy — the throughput criterion this
section argued for, at a small K:

| | ROCm | Vulkan |
|---|---:|---:|
| **generated token IDs** | `**Option` | **`**Option` — identical** |
| wall / tok | 418.2 ms | 728.1 ms |
| tok/s | 2.39 | 1.37 (**1.74x slower**) |
| **fetch hidden** | **97%** (270 ms fetched, 8 ms exposed) | **0%** (317 ms fetched, 566 ms exposed) |
| moe / tok | 311 ms (gpu 302 ms) | 566 ms (**gpu 0 ms**) |
| expert hit | 60.2% (238.5 miss/tok) | 60.5% (237.0 miss/tok) |
| pin build | — | 95.5 s |

Four readings, and the distinctions between them matter:

1. **The token IDs AGREE at K = 2.** Two full forward passes — 78 layers, a 154,880-way
   argmax, twice — landed on the same tokens. That is real evidence the port is
   numerically right, and it is not evidence that gate A passes at a useful K: the `exp`
   divergence this document measured makes eventual disagreement "a matter of when, not
   whether". K = 2 is a floor, not a result. **Measuring and reporting K is still owed.**
2. **0% fetch hidden against ROCm's 97%** is the single-queue prediction confirmed on
   matched runs, at matched hit rates (60.2% vs 60.5%, 238 vs 237 misses). Not a
   regression to chase — the three-queue design above is what removes it.
3. **1.74x slower, not ~3x.** This section predicted ~3x; the measurement at these
   settings is 1.74x, and the reason the prediction overshot is visible in the table: the
   exposed fetch is 566 ms while ROCm's whole token is 418 ms, so serialisation costs
   roughly one fetch pass rather than doubling everything. Recorded as measured-at-these-
   settings; a different hit rate moves it, and 1.74x should not be quoted as the
   backend's ratio.
4. **`gpu 0 ms` is the absence of a measurement**, not a fast kernel. See `vkstream.rs`.

The `--features vulkan,trace` build also decodes (2 tokens, exit 0), which exercises the
slot poisoning and the NaN localizer on this backend — and reported no non-finite layer,
a reading that only means something because `flag_nonfinite` was ported for real rather
than stubbed.

**What landed:** `backend.rs` became the real seam (launcher surface plus backend-neutral
`Stream`/`Event`/`Signal`); `gpu.rs`/`pin.rs`/`asyncfetch.rs`/`stream.rs` moved from
`#![cfg(rocm)]` to `any(rocm, vulkan)`; `src/vkstream.rs` is the single-queue shim;
`flag_nonfinite` was ported as a real shader and `fill_u32` as `vkCmdFillBuffer`; the 13
deferred kernels return `Err`, and `Config::validate_backend` refuses
`--mode int4|hybrid`, `--attn dsa|misa` and `--moe-gain != 1` at startup. The
host-vs-device pointer split landed in `ArenaPool` as this document said it must.

**Two findings from the run.** (1) is still open and now shows a THIRD message; (2) is fixed
by increment 2 — the staging copy is a `vkCmdCopyBuffer` on the fetch queue.

1. **`vkAllocateMemory` above ~4 GiB exceeds `maxMemoryAllocationSize` on this driver
   (4294967292 B), and the validation layer says so twice per run** — once for the
   16.16 GiB tier, once for the 43.8 GiB pool. Both allocations SUCCEEDED and the token
   decoded, which is precisely why this needs writing down: the limit is advertised, we
   exceed it by 11x, and the spec permits `VK_ERROR_OUT_OF_DEVICE_MEMORY` instead. Fixing
   it means suballocating `DeviceTier`/`VmmBuf` across several `VkDeviceMemory`
   allocations, which changes the address arithmetic `pin.rs`'s arena is built on — a
   design change, not a patch. Consequence in the meantime: any test allocating over 4 GiB
   will fail `Validation::check`, which asserts zero validation messages. See
   `vk::Buf::new`.
2. **Bounce mode's staging copy is synchronous on Vulkan.** `stream.rs`'s HIP path is an
   async `hipMemcpyAsync` on the fetch stream; the Vulkan path is a host `memcpy` into the
   host-coherent mapping, because the pool IS host memory here. It is correct and it costs
   a full un-overlapped copy per cold expert. `--direct-vmm-dma` skips it entirely by
   DMA-ing the O_DIRECT read straight into the mapping, which is what `VmmBuf::new`'s
   alignment guard exists for — worth measuring against the default on this backend, since
   the reason bounce is the ROCm default (a read tax on host-mapped VMM) may not price the
   same way when there is no H2D copy being saved.

   > **2026-08-01: that measurement was never taken, and the flag is now gone.**
   > `--direct-vmm-dma` was deleted and the staging hop is unconditional on both backends —
   > the amdgpu EFAULT it originally worked around no longer reproduces on 6.18.38, and on
   > ROCm the surviving objection is write-side and decisive (5.66 GB/s DMA into VMM device
   > pages vs 12.4 into the pinned arena; 1239 → 2709 µs per missed expert; 2.59 → 1.19
   > tok/s). **The Vulkan question this paragraph raises is genuinely still open** — the
   > pool is host memory here, so no H2D copy is being saved and the ROCm pricing may not
   > carry. It is now a question about re-adding a destination mode, not about flipping a
   > flag, and `src/fetch/stream.rs`'s header holds the measurement and the
   > `get_user_pages` history anyone re-opening it needs. Increment 2 fixed (2) by making
   > the staging copy a `vkCmdCopyBuffer` on the fetch queue, which is why nobody came back
   > to this.

**Two things this design must not quietly change.** Cross-queue submits make the
COMPUTE→COMPUTE and COMPUTE→TRANSFER barriers in `enqueue` insufficient on their own —
a barrier orders within a queue; a semaphore orders across. Every existing hazard the
barrier covers needs re-examining once a second queue can touch the same buffer, and
synchronisation validation on this stack **cannot see any of it** (it covers only
transfer↔transfer). That makes review the primary defence again, exactly as it was for the
original barrier. And the fetch queue writing a slot the compute queue is reading is the
`inflight` guard's job, not the queue layer's — the two must not both think they own it.

### Increment 2: measured. The overlap is back, and the residue is kernel speed.

Three queues, the ring, the async staging copy and the timestamp query pools, all landed
together because they are one design. **The invariant is now MEASURED rather than asserted**
— which was the whole reason the query pools moved forward out of phase 5.

**Matched runs**, same artifact (`/var/db/rivoli/glm52-vq3-full`), same flags
(`--mode int3-vq --attn dense --cache-policy lru --max-mem 115`), same prompt, greedy,
**`-bench 64`** — sixty-four tokens, not the 512 of the increment-1 table, so the two columns
here are comparable to each other and NOT to that table's absolute numbers (the hit rate is
still climbing at 64):

| | ROCm | Vulkan inc. 1 (512 tok) | **Vulkan inc. 2** |
|---|---:|---:|---:|
| tok/s | 2.73 | 1.44 | **1.46** |
| wall / tok | 366.1 ms | — | 685.9 ms |
| **fetch hidden** | **96%** (196 ms fetched, 8 ms exposed) | **0%** | **97%** (446 ms fetched, 15 ms exposed) |
| moe / tok | 255 ms (**gpu 246 ms**) | 522 ms (gpu 0 = absent) | 540 ms (**gpu 525 ms**) |
| route / tok | 97.8 ms | 128.2 ms | 113.3 ms |
| expert hit | 74.0% (155.8 miss/tok) | 76.5% | 74.6% (152.4 miss/tok) |
| ms / miss | 1.26 | — | 2.92 |
| pin build | — | — | 99.7 s |

The hit rates and miss counts match to under a point, which is what makes the rest of the
table a comparison rather than two runs.

**Four things the numbers say, and the distinctions matter as much as they did last time.**

1. **`fetch_hidden_pct` is a measurement now.** It reads 96%, against 0% in increment 1 and
   ~95-97% on ROCm. The 0% was an arithmetic artifact of `elapsed_ms` returning zero; this
   is `vkCmdWriteTimestamp` into a `VkQueryPool`, scaled by `timestampPeriod` and masked to
   `timestampValidBits`. Verified independently of the decode by
   `timestamps_measure_gpu_time_and_refuse_when_absent`: 0.661 ms of GPU for eight
   2048x2048 f32 GEMVs, i.e. 134 MB of weights at ~200 GB/s, inside a 0.948 ms wall.

   > **2026-08-01: half right, and the wrong half was the headline.** The `elapsed_ms` fix
   > IS real and the timestamps ARE measurements — that part stands. But
   > `fetch_hidden_pct` consumed them through `moe_wall − compute_gpu`, and `compute_gpu`
   > brackets the whole MoE phase including its stalls, so the ratio could not report
   > anything but ~97% regardless of what the timestamps said. Fixing the clock did not fix
   > the formula sitting on top of it. See `docs/reference/architecture.md` §3.

2. **The overlap invariant is upheld.** `moe/tok` is now almost entirely GPU time
   (`gpu` ≈ `moe`) — which, note, is exactly what an unfixed `compute_gpu` bracket reports
   whether or not it is true, so this bullet was never evidence for itself. The claim rests
   instead on increment 1 vs 2 at fixed everything-else. That is the architecture working:
   the reaper streams while the MoE
   queue computes.
3. **THE REMAINING GAP AGAINST ROCm IS NOT OVERLAP, IT IS KERNEL THROUGHPUT — and the GPU
   spans now prove it rather than suggest it.** The wall difference is 685.9 − 366.1 =
   319.8 ms/tok. The MoE GPU span alone accounts for 525 − 246 = **279 ms** of it, and
   `route` for another 15.5 ms; the exposed fetch contributes 7 ms. So ~87% of the gap is
   the MoE kernels being 2.1x slower than the HIP originals they were transliterated from,
   and essentially none of it is scheduling.

   Increment 1's `moe` +89% therefore had two causes stacked and only one was the queues,
   which is why "fix the queues and moe/tok halves" was never going to happen. This is the
   honest reading and it should not be dressed up: the port now implements the design, and
   the int3-vq shaders are slow. That is a kernel-optimisation question (occupancy, the
   12-bit gather's word-straddle path, `requiredSubgroupSize` interaction), it is now
   MEASURABLE per-phase, and it is not phase 4's scope.

   **A consequence for whoever does that work: the fetch is only just hidden.** 446 ms of
   fetch fits behind 540 ms of MoE with 15 ms exposed. Halve the MoE kernel time and the
   fetch stops fitting — it becomes the limiter, at 2.92 ms/miss against ROCm's 1.26 ms for
   the same bytes. The likely cause is the bounce arena: `Buf::staging` memory is
   host-visible but NOT page-locked, so every O_DIRECT read pays `get_user_pages` on ~3750
   pages that `hipHostMalloc`'s pre-pinned arena does not. It costs nothing today because it
   is hidden, and it will cost everything the moment the kernels get faster. Do not treat
   96% hidden as a permanent property of the backend.
4. **A fourth violation, which the checklist above did not have.** Fixing the queues is not
   enough on its own, because a Vulkan "launch" is a RECORD, not a submit. With one
   always-open command buffer, every MoE dispatch sat unsubmitted until the end-of-phase
   flush — so the GPU idled through the whole fetch and then computed. Fetch and compute
   stay serialised, in the other order, and the token-ID gate still passes. The MoE and
   fetch streams are therefore EAGER (one submit per launch, which is what HIP's launches
   already are) and the main stream stays lazy. Anyone porting this design to another
   backend should assume the same trap exists there: *ordering* the work correctly and
   *starting* it promptly are two properties, and only one of them is what a queue is for.

**The barrier review, and what carries each hazard.** Three mechanisms, none implying
another:

| hazard | execution order | availability | visibility |
|---|---|---|---|
| fetch copy → MoE dispatch reads the slot | host awaits the copy's `Signal` | that submit's fence + timeline signal | acquire barrier at the head of the MoE buffer |
| MoE reduce → main-queue `vadd` reads `moe_out` | host awaits `stream_signal(compute)` | same | acquire barrier at the head of the main buffer |
| main-queue writes `x` → MoE dispatch reads it | `copy_out_into`'s `device_sync` (the gate-logits D2H joins every queue before routing) | fence | acquire barrier |
| two batches on ONE queue (eager flush split them) | the barrier's first scope spans SUBMISSION order, across submits | — | same barrier |
| `fill_u32` poison → fetch copy writes the slot | `pin.rs`'s sync after the fill | fence | acquire barrier |
| slot RECYCLED while a dispatch reads it | `pin.rs`'s per-batch PIN SET — **not the queue layer** | — | — |

The last row is the one to keep straight: the queue layer owns the WRITE side of a slot and
the residency layer owns its LIFETIME. Two mechanisms both believing they own lifetime is
how a subtle double-free of bytes gets built.

> **CORRECTION.** The design text above (and "Two things this design must not quietly
> change") calls that mechanism "the `inflight` guard". **There is no `inflight` guard** —
> `grep -rn inflight src/` finds nothing but the comment that cites it. The real mechanism is
> the policy's per-batch pin set: `ArenaPool::begin_batch` clears it, phase 1a `protect`s
> every hit, phase 1b's `admit` pins every miss, and the end-of-layer `device_sync` closes
> the batch. So the property the doc asserted does hold, under a different name and by a
> different route. Recorded rather than quietly renamed, because the invented name is what a
> future porter reads first and it would send them looking for code that was never written.

The fourth row is the one that is easy to get wrong and silent when wrong: separate
submissions on one queue may otherwise execute concurrently, so the MoE reduce could read
partials mid-write. The barrier at the head of every command buffer is what prevents it, and
its being at the HEAD (not only between recorded ops) is load-bearing for exactly this.

Synchronisation validation sees none of this — it covers transfer↔transfer only — so the
tests carry what they can: `a_moe_dispatch_sees_what_the_fetch_queue_wrote` is the first row,
`a_signal_covers_recorded_work_without_a_device_sync` is the ring's contract, and
`the_command_buffer_ring_wraps_without_a_stale_slot` covers the wrap in both the
slots-still-pending and the across-joins regimes.

**`maxMemoryAllocationSize` is still exceeded, still deliberately, and now says so three
times.** Recorded, not fixed:

```text
vkAllocateMemory(): pAllocateInfo->allocationSize (17353937024) is larger than
maxMemoryAllocationSize (4294967292)
vkAllocateMemory(): pAllocateInfo->allocationSize (106126368768) is larger than
maxMemoryAllocationSize (4294967292)
vkAllocateMemory(): pAllocateInfo->allocationSize is 106126368768 bytes from heap 1,
but size of that heap is only 83393949696 bytes
```

The 16.16 GiB tier exceeds the advertised limit by 4x and the routed pool by 11x (98.8 GiB
at `--max-mem 115`); the third message is a different VUID — the pool also exceeds the HEAP
it is allocated from, because this APU reports a host-visible heap smaller than the memory
the driver will actually grant. All three allocations SUCCEED and the model decodes, which
is exactly why they are written down: the limits are advertised, the spec permits
`VK_ERROR_OUT_OF_DEVICE_MEMORY` for each, and a driver update is entitled to start returning
it. The fix is suballocating `DeviceTier`/`VmmBuf` across several `VkDeviceMemory`
allocations, which changes the address arithmetic `pin.rs`'s arena is built on — a design
change, not a patch. Consequence in the meantime: any test allocating over 4 GiB fails
`Validation::check`, and a Vulkan decode logs these errors per run. They are NOT spurious
and must not be filtered out.

## Risks

- **Synchronisation validation covers only transfer↔transfer on this stack. Every
  barrier in this backend is SPEC-DERIVED, NOT VERIFIED.** Measured, not assumed: with
  the barrier removed **entirely**, neither an unsynchronised compute→compute
  read-modify-write pair nor a compute-write → transfer-read produces a single
  message, while transfer↔transfer fires normally
  (the `vk_validation` probe's `compute-compute`, `compute-copy`, `compute-copy-desc`
  and `sync` modes; probe deleted, matrix kept in `docs/measurement/probes/README.md`, source at
  `77b5500:docs/measurement/probes/vk_validation`). It is not a buffer-device-address blind spot — a
  descriptor-bound write is equally invisible, so moving off bare device addresses
  would not buy coverage back.

  A compute backend is dispatches almost exclusively, so this is close to no coverage
  of the thing it is for.

  **The COMPUTE→COMPUTE barrier is nonetheless empirically validated, by demonstration
  rather than by a checker.** Deleting it makes
  `chained_dispatch_respects_the_barrier` fail; restoring it makes the same test pass:

  | configuration | barrier removed | barrier present |
  |---|---|---|
  | ×16 per run, 32 distinct matrices, 7.3 s | **8 of 8 runs FAIL** ← current | 8 of 8 pass |
  | ×16 per run, 1 reused matrix, 1.8 s | 1 of 8 FAIL | 8 of 8 pass |
  | ×1 per run, 32 distinct matrices, 7.4 s | 2 of 8 FAIL | 8 of 8 pass |
  | 4-step chain, 2048×2048 | 8 of 8 pass | 8 of 8 pass |
  | 2-step chain, 64×96 | 8 of 8 pass | 8 of 8 pass |

  Intermittent failure on removal with a clean control arm is a real race, so the
  barrier does observable work on this hardware. That is weaker than checker-verified
  and much stronger than spec-derived.

  Read the smaller rows too: at the sizes the test originally used, a **missing barrier
  is undetectable**. Anyone adding an ordering test needs enough workgroups to fill the
  machine several times over, every output depending on every input, and a chain long
  enough for one escaped hazard to survive to the end. Below that you get a test that
  passes either way — which is worse than none, because of the name on it.

  **Rows two and three are why the top row is worth its 7.3 seconds.** The middle row
  is what happened when the test was "optimised": collapsing 32 distinct matrices into
  one reused buffer cut runtime 7.4 s → 1.8 s and cut detection 2/8 → 1/8, and the
  sixteen repeats added in the same change did not compensate — they are correlated
  within a process, nowhere near the 1 − 0.75¹⁶ ≈ 99% the arithmetic suggests.

  The mechanism is memory traffic, not scheduling. 32 distinct 16 MB matrices make each
  step read cold DRAM; one reused matrix stays L2-resident, and the window closes. (It
  is NOT upload interleaving — every upload happens at setup, before the first
  dispatch.) Restoring the distinct matrices took detection to 8/8, because the repeats
  now compound on top of a window that is actually open.

  **The general lesson, which cost a real regression to learn: when optimising test
  scaffolding, ask what the scaffolding is doing for the test's SENSITIVITY.** Applied
  to production code, removing waste is free. Applied to a probe, the waste may be the
  instrument — and deleting it leaves every green still green while the test quietly
  stops being able to fail. The comment on the matrices in `tests/vk.rs` says this, so
  the next ponytail pass does not make the same correct-sounding argument.

  **Re-measure the removal arm after any change to this test.** The detection rate
  demonstrably does not follow from reasoning about it: 2/8, then 1/8, then 8/8, from
  changes that all looked neutral or positive beforehand.

  `memcpy_dtod_after_dispatch_is_ordered` has no equivalent construction at all, so
  COMPUTE→TRANSFER remains spec-derived only.

  **Second, independent evidence that this cover is thin.** The Phase 2 correctness
  review found that `DeviceBuf`'s copies did not synchronise where `hipMemcpy` blocks —
  at integration, `gpu.rs:789` would have read the *previous* token's gate logits and
  mis-routed every expert on every layer. That is squarely the class synchronisation
  validation exists to catch, and it was found by human review of the diff, not by any
  checker. Two independent routes to the same conclusion: **on this stack, review is
  the primary defence for ordering bugs and the layer is a secondary one.** Budget
  Phase 3 accordingly — a ported kernel's clean validation run is not evidence that its
  synchronisation is right, and the oracles do not test ordering either.
- **Subgroup size control** may be unavailable or ignored on some drivers; a wave64
  fallback means re-tuning `ROWS_PER_BLOCK` and re-validating determinism. Note the
  concrete shape of that failure: `gemv_f32.comp` maps rows by `gl_SubgroupID`, so a
  wave64 subgroup halves `gl_NumSubgroups` and rows 4..7 of every workgroup are simply
  never written — stale contents, no fault, no diagnostic. Only an oracle catches it.
- ~~No validation layer on this box~~ **RESOLVED.** `VK_LAYER_KHRONOS_validation`
  1.4.341 is installed. `Gpu::validation_layer` enables it automatically whenever the
  loader advertises it, so **the suite runs under validation by default** — there is no
  longer an opt-in step to forget. To get an un-validated baseline, force it off with
  `VK_LOADER_LAYERS_DISABLE='*'` (confirmed supported by this loader).

  The opt-in checkers are the part that is easy to get wrong, because the default
  config runs neither and a clean run under it says nothing about either:

  | check | how | covers |
  |---|---|---|
  | core (VUIDs) | on by default | API misuse, object lifetimes |
  | synchronisation | `VK_LAYER_VALIDATE_SYNC=1` | **the `Gpu::enqueue` barrier**, hazards |
  | GPU-assisted | `VK_LAYER_GPUAV_ENABLE=1` | shader-side buffer-device-address reads |

  Use the modern env vars, not `khronos_validation.enables` in a settings file — the
  deprecated key makes the layer emit two configuration warnings of its own, which
  `assert_validation_clean` then counts as findings. Config noise must not look like a
  result. Run GPU-AV in its own pass: with core validation also on, the layer warns
  that the combination is slow, and that warning is likewise counted.

  Status as of the commit carrying this note: the whole suite, **37 tests**, run once
  under each of the three configurations in the table above (GPU-AV in its own pass):
  **all clean.** That is coverage evidence, not repeatability evidence — no repeats were
  taken. **It does not read back onto earlier commits:** `bffdc9d` is RED on
  `attend_honours_the_dsa_row_selection`, which shipped a fixed-point precondition and
  data violating it in the same commit, so the suite had never been green at 37 before.

  **GPU-AV was confirmed live by DEMONSTRATION, not by its self-report:** the
  `vk_validation` probes were re-run on this stack the same day and reproduce the
  matrix below exactly (probe since deleted — `77b5500:docs/measurement/probes/vk_validation`). (The self-report — 2 `VALIDATION-SETTINGS`-class messages with
  the env var, 0 without — establishes only that the layer read the variable.)

  **The `down` path at `inter = 64` is clean under GPU-AV *at that shape*.** That is the
  adversarial shape for `bffdc9d`'s scale-read fix, which had never executed before this
  run. Three limits, because this is otherwise exactly the paragraph that gets quoted as
  a licence to stop checking:

  - `bffdc9d` fixed **two** defects and GPU-AV can only ever have seen one. The 2-byte
    over-read leaves the buffer, so it is in scope; the **misaligned 32-bit load is
    in-bounds and has no instrument here**.
  - The buggy code never ran under GPU-AV, so nobody watched the checker fire on this
    defect. A clean post-fix pass is *consistent with* the fix and is not evidence the
    checker would have caught it.
  - The defect class is numerically silent, so GPU-AV's silence covers exactly the shapes
    that were run. Re-run this shape after any change to the scale read.
- ~~GPU-AV's buffer-address checker has not been proven to fire~~ **CLOSED.** All three
  checkers have now been observed catching a deliberate fault, so their silence is
  evidence rather than an absence of evidence:

  | checker | fault used to prove it | reported |
  |---|---|---|
  | core | `vkCreateBuffer(size = 0)` | `VUID-VkBufferCreateInfo-size-00912` |
  | synchronisation | two `vkCmdFillBuffer`s, no barrier | `SYNC-HAZARD-WRITE-AFTER-WRITE` |
  | GPU-assisted | OOB store through a buffer reference | `VUID-RuntimeSpirv-PhysicalStorageBuffer64-11819` — "Out of bounds access: 4 bytes written at buffer device address 0x…" |

  The GPU-AV proof matters most: it is the only checker that can see a bad *device
  address*, because the address is an opaque `uint64` in a push constant with no object
  for the CPU side to bounds-check against.

  **And it earned that immediately.** On the first run of the `fwd.hip` tranche under
  GPU-AV, `embed_i8_row` reported
  `Out of bounds access: 4 bytes read at buffer device address 0x…`, invocation 36 of a
  5 × 37 = 185-byte embedding table reading the word at offset 184 — three bytes past
  the end. Cause: shaders read packed `u8` as 32-bit words (`VK_KHR_8bit_storage` is
  deliberately not required), so the word holding an unaligned buffer's last byte
  overruns it.

  **Every numeric oracle passed.** In the default configuration, and under
  synchronisation validation, and on the byte-exact comparisons — because the
  out-of-range bytes land in the discarded lanes of the word. Nothing but GPU-AV could
  see it, and GPU-AV was only run because it had first been *proved to fire* against a
  deliberate fault. That is the entire argument for the probe exercise, in one incident.

  Scope, because the distinction matters when assessing this backend: the real engine's
  embedding table is 151552 × 5120 = 776,048,640 bytes, which **is** divisible by 4, so
  the bug was latent there and live only in the test's ragged shape. We found one before
  it could exist rather than shipping one — but only because the oracle used a
  deliberately ragged `hidden = 37`, and only because it ran under a checker known to
  work.

  Fixed by `vk::Buf::new` rounding every allocation up to `WORD`, reporting the unpadded
  `len` so no downstream bounds check starts permitting real overrun. That is the fix;
  `DeviceTier::place`'s matching cursor padding is belt-and-braces and an earlier
  version of this section overstated it as load-bearing. It is not: `place` already
  rounds offsets to 256, so the padding changes no address today. It is kept so the
  invariant rests on the padding rather than on 256 happening to exceed a word — a
  future tightening of that alignment would otherwise turn a benign read into a live
  overrun between placements.

  No launcher assert: `launch_embed_i8_row` never receives a length, and the nearest
  checkable condition (`hidden % 4 == 0`) would have banned the ragged oracle that found
  the bug.

  Note that `VALIDATION-SETTINGS` and `WARNING-Setting-Limit-Adjusted` are the layer
  describing its OWN configuration (GPU-AV forcing `vulkanMemoryModel` on so it can
  instrument, and warning that core+GPU-AV together are slow). `debug_callback` logs
  them but excludes them from `VALIDATION_ERRORS` by exact message-ID match — otherwise
  the suite could never pass under GPU-AV, which is the checker that matters most here.
- **Push-constant budget** is 128 bytes guaranteed. Buffer device addresses collapse
  every buffer argument to 8 bytes, which may retire this risk outright: `gemv_f32`
  fits in 32 bytes with zero descriptor sets, and the worst case — `moe_gateup_vq`
  with `ExpertDesc`'s six pointers plus three codebooks plus dims — is ~88 bytes if
  the descs stay a pointed-to array. Confirm at `launch_attend` and the MoE launchers
  and record the measurement here; do not pre-emptively build the params-buffer path.
- **Profiling**: no `rocprof` equivalent in the workflow, but timestamp query pools are
  strictly better than the current HIP-event bracketing — the existing `telemetry.rs`
  spans map over cleanly.
- **Scope**: 16 kernels is still not a weekend, and the deferred 13 are deferred, not
  cancelled — `--mode int4|hybrid` and `--attn dsa` remain HIP-only until someone ports
  them. That is a smaller capability set for that build, not a mixed backend: the HIP
  build keeps everything, and neither build ever calls into the other.
