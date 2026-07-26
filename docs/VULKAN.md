# rivoli — Vulkan backend (`vulkan` feature)

Status: **proposed, not started.** This is the plan, not a description of code that
exists. See [ARCHITECTURE.md](ARCHITECTURE.md) for the engine it plugs into.

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

Fail fast at init with a clear message naming the missing one:

- `VK_KHR_buffer_device_address` (core 1.2) — lets `ExpertDescVq`'s six raw pointers
  stay six `uint64` addresses in a params buffer instead of six descriptor bindings.
  Without it the per-expert launch path needs a descriptor rewrite per expert.
- `shaderInt64` (core 1.0) — the other half of the above: the shader dereferences
  those addresses as `uint64_t` buffer references (`GL_EXT_buffer_reference`), which
  is a 64-bit integer operation.
- `VK_EXT_subgroup_size_control` with `requiredSubgroupSize = 32` — the kernels assume
  `WAVE 32` (gfx1151 native wave32; see `kernels/common.hpp`).
- `subgroupShuffleRelative` + `subgroupBasic` (core 1.1 subgroup ops).
- `VK_KHR_shader_float16_int8` + `VK_KHR_16bit_storage` — the fp16 VQ codebook.
- `VK_KHR_timeline_semaphore` (core 1.2).
- A `DEVICE_LOCAL | HOST_VISIBLE` heap covering GTT — on Strix Halo host RAM *is* GPU
  memory, and the pin path depends on writing resident weights directly.

Deliberately **not** required: `VK_KHR_8bit_storage`. Read packed u8 weights as `uint`
words and unpack bytes/nibbles in the shader — colibri's `qmatmul.comp` does exactly
this, and the hot loops already unpack manually, so the extension buys nothing.

Use a **dedicated compute queue**, not the universal graphics queue (colibri `4d4cacc`).

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

## Kernel inventory — port 16 of 29

v1 is **`--mode int3-vq`, `--attn dense`, decode only**. That is the smallest thing that
runs the model, and it cuts 13 kernels from the port.

| File | Port now | Port difficulty |
|---|---|---|
| `linalg.hip` | `gemv_fp8`, `gemv_fp8_splitk`, `gemv_f32`, `gemv_i8`, `swiglu`, `rmsnorm`, `rope_interleave` | `gemv_f32`/`swiglu`/`rmsnorm` trivial; the split-K LDS combine and the e4m3 LUT need care |
| `moe.hip` | `moe_gateup_vq`, `moe_down_vq`, `moe_reduce` | hardest — per-expert device addresses, fp16 codebook gather |
| `fwd.hip` | `embed_i8_row`, `append_kv`, `gather_rope`, `vadd`, `argmax_reduce` | easy |
| `mla.hip` | `mla_absorb_fp8`, `mla_value_fp8` | easy-moderate |
| `attn.hip` | `mla_latent_attend`, `mla_attend_combine` | moderate — dynamic shared memory sizing becomes a specialization constant |

**Deferred, with the reason:**

- `vq_encode` (`linalg.hip`) — converter only. `convert --gpu` stays a ROCm-build tool;
  a Vulkan box converts on a HIP machine or on CPU.
- `gemv_vq`, `gemv_i4` (`linalg.hip`) — standalone microbench/oracle kernels. Decode
  goes through `moe_*_vq`; nothing in the forward pass calls these.
- `moe_gateup_i4`, `moe_down_i4` (`moe.hip`) — only needed for `--mode int4|hybrid`.
- `indexer.hip` ×5 — the DSA path. A Vulkan build supports `--attn dense` and rejects
  `--attn dsa` at startup.

`vmm.hip` and `async.hip` are runtime shims, not kernels — replaced by the Vulkan
memory/queue layer rather than ported.

**Shader language: GLSL compiled by `glslc`.** The kernels are already C-like; GLSL
keeps the diff readable against the `.hip` originals, which matters because the two
must stay numerically identical. Slang would also work but adds a toolchain nobody
here has. `build.rs` gains a `vulkan` arm compiling `kernels/vk/*.comp` → SPIR-V,
embedded via `include_bytes!`. Keep `rerun-if-changed` on the shared header — the
`common.hpp` staleness bug (see git history) is easy to repeat with `#include`d GLSL.

### Numerics that must stay bit-exact

### Same spelling, different semantics — the porting pre-flight

**Run this list against the HIP source BEFORE writing a shader, not after the oracle
fails.** Every entry below was discovered the expensive way, and each one was silent at
every stage before a numeric comparison. They are one class: a construct that is spelled
the same or nearly the same in HIP and GLSL, and means something different. That class is
where this port's real defects live — not in the algorithms, which transliterate fine.

| HIP | GLSL | Mechanism | Symptom |
|---|---|---|---|
| `float* lut` parameter | `float lut[256]` parameter | C decays arrays to pointers; GLSL copies value-result | Per-invocation private copy, writes lost, silent garbage. Rule 11 |
| `i / block`, signed | `i / block` | LLVM strength-reduces, but the signed-quotient correction survives | Silent cost — 44→29 VALU/iter. Use `findMSB` |
| `__shfl_down(v, o, 32)` | `subgroupShuffleDown` | Out-of-range returns the caller's own value in HIP; UNDEFINED in SPIR-V | Silent divergence on the last lanes. Rule 2 bans it |
| `1.0f/sqrtf(z)` | `inversesqrt(z)` | Different accuracy contracts under identical intent | Silent ULP divergence from the CPU oracle. Rule 9 |
| `exp2f(float(k))` | `exp2(float(k))` | Exact on an integer argument in libm; NOT REQUIRED to be in GLSL | Silent ULP error under a claimed bit-exactness contract |
| `(unsigned char)p` | *(no such cast)* | C truncates; GLSL has no 8-bit type, so it must be `p & 0xFFu` | A wrong mask is a wrong weight, not a compile error |
| `isfinite(x)` | *(does not exist)* | Must be composed from `isinf`/`isnan` | Compiles only if you notice; easy to get the polarity backwards |
| `size_t` index | `uint` index | 32-bit overflow at 4 GB in the shader, not on the host | Silent wrap on large tensors. See the Index width note |
| `OpFMul` + `OpFAdd` | *one* fused multiply-add | The DRIVER contracts below SPIR-V. The module still shows two opcodes | The disassembly does not tell you the arithmetic. See below |

**The FMA row is the one that breaks the instrument.** Every other entry is caught by
reading the source or the disassembly; this one is invisible in both. `glslc` emits
`OpFMul` then `OpFAdd`, the module says so, and the driver's backend fuses them anyway —
so the single-rounding FMA result differs from the two-rounding result, and nothing you
can inspect statically says which you got.

Measured, and it is not marginal: a CPU oracle modelling `acc += s * (a+b+c+d)` as written
differed from `mla_value_fp8` on **10 of 15 outputs**; modelling the fused chain
(`t = x0*l0; t = fma(x1,l1,t); …; acc = fma(s,t,acc)`) made it **exactly zero**.

Two consequences worth carrying:

- **"Disassemble anything whose bit-exactness matters" is necessary and not sufficient.**
  That advice appears twice in this document and it was written before this was known. The
  disassembly is ground truth for *which operations exist*, not for *how they are rounded*.
- **It is probably why the two backends agree.** hipcc defaults to contracting too
  (`-ffp-contract=fast` is on at `-O3` without `-ffast-math`), so HIP and Vulkan plausibly
  fuse the same expressions. That is a reason the port works, not a reason to relax: it
  means bit-identity between backends rests on two independent compilers making the same
  contraction choice, which is an assumption nobody has stated until now.

The generative rule, which is worth more than the table: **a HIP construct whose GLSL
analogue is "the obvious builtin" needs an explicit decision, never a transliteration.**
That is exactly how `inversesqrt` got in, and how `exp2` sat under a comment calling
bit-exactness a contract until this list was written.

### THE EXACTNESS STRATEGY: eliminate the degree of freedom, do not hope it is exercised alike

The single most consequential fact in this port, and it took until the FMA finding to state
plainly:

> **Cross-backend bit-identity rests on two independent compilers making the same choice
> wherever their specifications leave one open.** Careful porting does not achieve
> exactness — it only makes the two sides ELIGIBLE to agree. The agreement itself is
> contingent on decisions neither backend's author controls.

Every accuracy contract (`exp2` at 3 ULP, `inversesqrt` at 2 ULP), every latitude to
contract a multiply-add, every unspecified summation order is such a choice. Two conformant
compilers may resolve each differently and both be right.

**So the strategy is not to match the choices. It is to remove them.** Three fixes already
in this tree are instances of one move, and naming the move matters more than the list,
because the list cannot tell you what to do about a case it does not contain:

| degree of freedom | how it was removed |
|---|---|
| `exp2`'s accuracy contract | power of two **built from bits** — exact by construction, no contract left to differ on |
| `pow`/`cos`/`sin` in rope | evaluated **host-side in f64**, uploaded as a table — no shader transcendental left to differ on |
| which multiply fuses in `a*b + c*d` | a **named temporary** for one product — no contraction freedom left to differ on |

The pattern: find the point where the two toolchains are permitted to disagree, and
restructure so the permission never arises. That is stronger than testing for agreement,
because a test can only observe today's compilers.

**When you meet a new case, ask in this order:**

1. **Can the value be constructed exactly?** (`exp2` — integer argument, exact power of two.)
2. **Can the work move to a place with one implementation?** (rope — the host, in f64.)
3. **Can the expression be written so only one lowering is legal?** (the accumulator — split
   the products across statements.)
4. Only if none apply: **pre-register the divergence**, name the mechanism, and put it on
   the token-ID gate's suspect list. `exp` in softmax is the one genuine case so far, and it
   is genuine because its argument is an arbitrary runtime value.

**Why this hazard class appears in 2c and not earlier, which also predicts where it appears
next.** Contraction only has a choice to make when TWO multiplies feed one add. Every
kernel ported through 2b has at most one multiply per add — `acc += x*w`, `acc += s*(…)` —
so both compilers fused the only candidate and the backends agreed *by having no freedom*,
not by anyone's care. `attn.hip:185`'s `acc[k]*corr + p*Lt[..]` is the first expression in
the port with two, and the first place the choice is real. Grep for that shape when
approaching any remaining kernel.

### A WORK PARTITION IS A NUMERICS DECISION. `mla_plan_splits` IS NOT A PERF KNOB.

The one divergence class that lives BETWEEN kernels rather than inside one, which puts it
outside the reach of every instrument this port has built.

`attn.hip::mla_plan_splits` cuts the `nr` attended rows into `n_splits` chunks. It reads
like tuning — it is derived from CU count and a work threshold, and its own comment
describes occupancy. But the cut determines **which rows each split reduces**, therefore
the summation order, therefore the bits. The HIP source says so in passing ("same context
length → same cut → same summation order → bit-identical results"); the Vulkan port has to
reproduce the function EXACTLY, not merely plausibly.

**Why no existing instrument would catch a divergence here.** Each backend's per-kernel
oracle models that backend's own plan, so each agrees with its own reference and both go
green. The numbers differ only when the two are compared to each other — which nothing does
until the token-ID gate. So a "harmless" retuning of either copy — bumping
`MLA_TARGET_BLOCKS` for a new GPU, changing `MLA_MIN_TILES_PER_SPLIT` — is a **silent
numerics change** that presents as a decode divergence months later, in a different file
from the one that was edited.

Guarded by `tests/vk.rs`'s attention oracle asserting a specific split count for its shape,
so a drift in either copy fails loudly and locally. That is a weak guard for a strong
hazard; it is what is available without a cross-backend harness.

**The general shape, worth carrying to the remaining kernels:** *anything that partitions
work and feeds an order-sensitive reduction is a numerics decision wearing a scheduling
costume.* Candidates already in this codebase: `gemv_fp8`'s split-K threshold (a bare 4096
literal on the Vulkan side, uncoupled from the HIP constant — the same defect, unfixed),
`moe_reduce`'s expert batching, and any future change to `TILE` or `HB`. Ask of each: if I
retuned this for speed, would the OUTPUT change? If yes, it is single-sourced or it is a
bug waiting for a profiler.

Two of these have mechanised guards (rules 2 and 9) and one is preventive (rule 11).
The rest are checked by reading, which is why the list exists.

### Rule 11 is the first PREVENTIVE rule

Every other mechanised rule is detective: it fires on a defect that already exists in the
tree. Rule 11 fires on a defect about to be introduced. `mla_absorb_fp8` and
`mla_value_fp8` both call `e4m3_lut_build`, so a mechanical transliteration of tranche 2b
reintroduces the copy-in/copy-out bug TWICE, in two kernels whose oracles do not exist
yet — and it would again be silent at every stage before the numeric check. The rule was
written an hour before the code it will stop.

That is the strongest available argument for mechanising a rule the moment it is
understood rather than after the next instance.

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

`bf16f`/`f2bf16`/`e4m3f` in `common.hpp` are bit-exact with `src/math.rs` **on the
finite domain** — the CPU oracles in `tests/kernel.rs` test that. They are NOT
bit-exact for NaN: `half::bf16::from_f32` forces the quiet bit (`| 0x0040`) where
`common.hpp`'s `f2bf16` returns the top 16 bits verbatim. That divergence predates
this port; a GLSL version must mirror **HIP**, not `math.rs`, or the two backends
disagree. GLSL has no `__builtin_memcpy`; use `floatBitsToUint`/`uintBitsToFloat`. The
e4m3 LUT (`e4m3_lut_build`) ports directly to a 256-float `shared` array — and prefer
the LUT to a live `exp2` call, since GLSL specifies `exp2` only to 3 ULP.

**Index width.** GLSL `uint` row arithmetic wraps at 2^32 elements = 4.29e9. `gemv_f32`
(router gate, 256x5120 = 1.3e6) is nowhere near it, but `lm_head` via `gemv_i8` is
151552x5120 = 7.76e8 — only 5.5x of margin, and it would wrap SILENTLY into another
allocation. Use `uint64_t` row indices from `gemv_i8`/`gemv_fp8`/`moe_*` onward
(`GL_EXT_shader_explicit_arithmetic_types_int64` is already enabled and `shaderInt64`
already required) and record the ceiling per kernel as you port.

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
`bin/ppl` already has, and the same verdict vocabulary already in `benchmarks.md`
(PASS / FAIL / COST ESTABLISHED / INCONCLUSIVE).
*Catches:* systematic accuracy loss that a short token-ID run would miss entirely, and it
is the only part that speaks to output QUALITY rather than agreement.
*Cannot catch:* a rare catastrophic path the corpus never exercises; and it is
underpowered at small n — `benchmarks.md` already records three of four cells being unable
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

#### What I would actually defend

**Accept on A reported + B within bound + C within bound, with D green.** A is a number in
the release notes rather than a threshold; B and C are the two pass/fail criteria; D is the
regression net that runs on every commit.

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

This is the fourth failure class from `docs/probes/README.md` — *a check constitutionally
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
`benchmarks.md`'s "Read the ISA before you book the device" prescribes. It also would not
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

(This table belongs in `benchmarks.md` at merge time — it is kept here for now to avoid
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
5. **Bench + profiling** — timestamp query pools wired into `telemetry.rs`, then compare
   against HIP on the same artifact. Expect the first port to be slower; record where,
   do not tune during the port.

**Sequence this last.** Of the three open proposals it has the least measured upside:
PILOT and `top-m` attack a bottleneck we have quantified, while this buys portability
we do not currently need on a node where ROCm works. Build it when the portability goal
is real, not because the plan exists.

## Phase 4 needs two queues, not one — and the acceptance gate cannot see it

Written before integration starts, because the failure it describes is invisible to every
check this port has: **the token-ID gate passes a backend that is three times slower.**

### The streaming layer is not ported, and has not strayed only because it does not exist

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

### The performance property lives in the CONCURRENCY, and vk.rs has none

The engine's headline behaviour — fetch ~95% hidden behind compute (`benchmarks.md`) —
comes from **two independent streams**:

| piece | where |
|---|---|
| the reaper's dedicated fetch stream | `asyncfetch.rs`, `HipStream::new()` in `AsyncFetch::new` |
| the MoE expert compute stream | `gpu.rs`, `compute_stream`, explicitly "separate from the null stream the rest of the forward uses" |
| the driver that races them | `gpu.rs`, `try_for_each_concurrent` over the expert descriptors |

`vk.rs` today is **one queue behind one `Mutex<Cmd>`**, documented as mirroring "HIP's
default stream". Integrating onto that serialises fetch against compute.

**And the acceptance gate would not notice.** Token IDs depend on arithmetic, not on
overlap; a fully serialised backend computes exactly the same numbers. So the gate this
document spent pages sharpening — identical token IDs, greedy decode, pinned conditions —
passes a build whose central performance property has collapsed. **Phase 4 therefore needs
a throughput criterion alongside it: measure fetch-hidden % and tok/s, and compare against
the ROCm numbers in `benchmarks.md` at matched `--max-mem` and `--cache-policy`.** A
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

**Two things this design must not quietly change.** Cross-queue submits make the
COMPUTE→COMPUTE and COMPUTE→TRANSFER barriers in `enqueue` insufficient on their own —
a barrier orders within a queue; a semaphore orders across. Every existing hazard the
barrier covers needs re-examining once a second queue can touch the same buffer, and
synchronisation validation on this stack **cannot see any of it** (it covers only
transfer↔transfer). That makes review the primary defence again, exactly as it was for the
original barrier. And the fetch queue writing a slot the compute queue is reading is the
`inflight` guard's job, not the queue layer's — the two must not both think they own it.

## Risks

- **Synchronisation validation covers only transfer↔transfer on this stack. Every
  barrier in this backend is SPEC-DERIVED, NOT VERIFIED.** Measured, not assumed: with
  the barrier removed **entirely**, neither an unsynchronised compute→compute
  read-modify-write pair nor a compute-write → transfer-read produces a single
  message, while transfer↔transfer fires normally
  (`docs/probes/vk_validation`: `compute-compute`, `compute-copy`,
  `compute-copy-desc`, `sync`). It is not a buffer-device-address blind spot — a
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

  Status as of the first validated run: **core and synchronisation validation both
  clean across all four tests.** GPU-AV clean too, but see the caveat below.
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
