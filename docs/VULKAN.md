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
| `hipHostMalloc` (bounce arena) | `VK_EXT_external_memory_host` import, or mapped `HOST_VISIBLE` |
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
- `VK_EXT_subgroup_size_control` with `requiredSubgroupSize = 32` — the kernels assume
  `WAVE 32` (gfx1151 native wave32; see `kernels/common.hpp`).
- `subgroupShuffleRelative` + `subgroupBasic` (core 1.1 subgroup ops).
- `VK_KHR_shader_float16_int8` + `VK_KHR_16bit_storage` — the fp16 VQ codebook.
- `VK_KHR_8bit_storage` — packed u8 weights (or read as `uint` and unpack, which the
  hot loops already do).
- `VK_KHR_timeline_semaphore` (core 1.2).
- A `DEVICE_LOCAL | HOST_VISIBLE` heap covering GTT — on Strix Halo host RAM *is* GPU
  memory, and the pin path depends on writing resident weights directly.

## Two things that are not mechanical

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

## Kernel inventory (29 compute kernels)

| File | Kernels | Port difficulty |
|---|---|---|
| `linalg.hip` | `gemv_fp8`, `gemv_fp8_splitk`, `gemv_vq`, `gemv_f32`, `gemv_i8`, `gemv_i4`, `swiglu`, `rmsnorm`, `rope_interleave`, `vq_encode` | `gemv_f32`/`swiglu`/`rmsnorm` trivial; the split-K LDS combine and the e4m3 LUT need care |
| `moe.hip` | `moe_gateup_vq`, `moe_down_vq`, `moe_reduce`, `moe_gateup_i4`, `moe_down_i4` | hardest — per-expert device addresses, fp16 codebook gather |
| `fwd.hip` | `embed_i8_row`, `append_kv`, `gather_rope`, `vadd`, `argmax_reduce` | easy |
| `mla.hip` | `mla_absorb_fp8`, `mla_value_fp8` | easy-moderate |
| `attn.hip` | `mla_latent_attend`, `mla_attend_combine` | moderate — dynamic shared memory sizing becomes a specialization constant |
| `indexer.hip` | `layernorm`, `index_append`, `index_score`, `index_pool_push`, `index_head_route` | moderate; DSA path, defer to last |

`vmm.hip` and `async.hip` are runtime shims, not kernels — they are replaced by the
Vulkan memory/queue layer rather than ported.

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
that suite passes unmodified**, plus a coherent 32-token decode on a partial artifact
(the `convert --layers 1` stub path used for the HIP end-to-end bring-up).

Do not accept "close enough" numerics. The oracle tolerances were tuned against real
failures — bf16 codebooks failed the 1e-3 oracle and fp16 passed with ~5.6× margin;
that margin is the safety net for a second backend.

## Staging

Each phase ends with something runnable; no phase leaves the tree broken.

1. **Spike** — instance/device/queue init, feature detection, one kernel (`gemv_f32`)
   dispatched from a GLSL shader, its oracle test green under `--features vulkan`.
   Proves the waist and the build plumbing. Small; do it before committing to the rest.
2. **Memory + sync** — `DeviceBuf`/`DeviceTier`/`VmmBuf` equivalents, timeline-semaphore
   `Signal`, timestamp queries, host-pointer import for the io_uring bounce arena.
   Exit: `DeviceTier` reserve/copy tests pass; the reaper's H2D path round-trips.
3. **Kernels, oracle-first** — port in test order: `fwd` → `linalg` (non-MoE) → `mla` →
   `attn` → `moe`. Exit: full `tests/kernel.rs` green.
4. **Integration** — `backend.rs` switch, `pin.rs`/`gpu.rs` imports, end-to-end decode
   on the 4-layer stub, then the full artifact. Exit: coherent output.
5. **Bench** — compare against HIP on the same artifact. Expect the first port to be
   slower; record where, do not tune during the port.

## Risks

- **Subgroup size control** may be unavailable or ignored on some drivers; a wave64
  fallback means re-tuning `ROWS_PER_BLOCK` and re-validating determinism.
- **Host-pointer import alignment** (`minImportedHostPointerAlignment`) must be ≤ the
  O_DIRECT 4096 alignment the streamer already guarantees; check at init.
- **Push-constant budget** is 128 bytes guaranteed. `launch_attend` and the MoE
  launchers exceed that — those need a small params `VkBuffer` per dispatch, which is
  an extra write on a hot path. Measure before assuming it is free.
- **Profiling**: no `rocprof` equivalent in the workflow, but timestamp query pools are
  strictly better than the current HIP-event bracketing — the existing `telemetry.rs`
  spans map over cleanly.
- **Scope**: 29 kernels is not a weekend. If the goal is only "runs without ROCm", the
  first Vulkan release can ship without the DSA indexer path (5 kernels) — a Vulkan
  build would support `--attn dense` only and reject `--attn dsa` at startup. That is a
  smaller capability set for that build, not a mixed backend: the HIP build keeps DSA,
  and neither build ever calls into the other.
