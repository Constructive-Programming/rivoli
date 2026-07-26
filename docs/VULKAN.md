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

## Three things that are not mechanical

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
change. Measure the latency — a thread wakeup may beat 19 µs or may not; if it
regresses, batch several completions per semaphore value.

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

`bf16f`/`f2bf16`/`e4m3f` in `common.hpp` are bit-exact with `src/math.rs` by
requirement — the CPU oracles in `tests/kernel.rs` test exactly that. GLSL has no
`__builtin_memcpy`; use `floatBitsToUint`/`uintBitsToFloat`. The e4m3 LUT
(`e4m3_lut_build`) ports directly to a 256-float `shared` array.

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

Scoped, not "unmodified", on purpose: the suite also covers `gemv_vq`/`gemv_i4`, which
are microbench kernels the decode path never calls. Gate those oracles on the ported
set (`#[cfg]` or a skip list) rather than porting two kernels to satisfy a test.

Do not accept "close enough" numerics. The oracle tolerances were tuned against real
failures — bf16 codebooks failed the 1e-3 oracle and fp16 passed with ~5.6× margin;
that margin is the safety net for a second backend.

## Staging

Each phase ends with something runnable; no phase leaves the tree broken.

1. **Spike** — instance/device/queue init, feature detection, one kernel (`gemv_f32`)
   dispatched from a GLSL shader, its oracle test green under `--features vulkan`.
   Proves the waist and the build plumbing. Small; do it before committing to the rest.
2. **Memory + sync** — `DeviceBuf`/`DeviceTier`/`VmmBuf` equivalents, timeline-semaphore
   `Signal`, and the device-local copy pool for the io_uring bounce path. No profiling
   yet. Exit: `DeviceTier` reserve/copy tests pass; the reaper's H2D path round-trips.
3. **Kernels, oracle-first** — port in test order: `fwd` → `linalg` (non-MoE) → `mla` →
   `attn` → `moe`. Exit: the 16 ported kernels' oracles green.
4. **Integration** — `backend.rs` switch, `pin.rs`/`gpu.rs` imports, end-to-end decode
   on the 4-layer stub, then the full artifact. Exit: coherent output.
5. **Bench + profiling** — timestamp query pools wired into `telemetry.rs`, then compare
   against HIP on the same artifact. Expect the first port to be slower; record where,
   do not tune during the port.

**Sequence this last.** Of the three open proposals it has the least measured upside:
PILOT and `top-m` attack a bottleneck we have quantified, while this buys portability
we do not currently need on a node where ROCm works. Build it when the portability goal
is real, not because the plan exists.

## Risks

- **Subgroup size control** may be unavailable or ignored on some drivers; a wave64
  fallback means re-tuning `ROWS_PER_BLOCK` and re-validating determinism.
- **No validation layer on this box** (standing, unmitigated). `VK_LAYER_KHRONOS_validation`
  is not installed on the gfx1151 node and installing it needs root
  (`media-libs/vulkan-layers` on Gentoo). This is the worst gap in the port: passing
  every buffer as a bare device address with no descriptor sets means a wrong address
  or a missing barrier reads plausible **garbage** rather than faulting, and that is
  exactly the class the layer exists to catch. What we have instead: `spirv-val` on
  every module in `build.rs` (static wellformedness only — it sees nothing about
  synchronisation, descriptors, or BDA), and the numeric oracles. Init logs at WARN
  when the layer is absent so a green run is never mistaken for a validated one.
  Re-run the whole suite under the layer the moment it is installed.
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
