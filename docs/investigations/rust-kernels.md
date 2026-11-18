---
status: live
scope: engine
verdict: OPEN 2026-11-18 — migrate every HIP C++ kernel (crates/backend/kernels, 15 .hip + 8 .hpp, 8115 lines, 66 launchers) to Rust on amdgcn-amd-amdhsa, owner decision for a smaller unified codebase. The spike proved bit-identity on two kernels and named three hazards (FMA contraction, missing workgroup-scope fence, an unexplained ~1 µs small-launch gap); this plan is a hypothesis, and a measured reason to stop is a valid outcome.
---

# Rust kernels — migration plan

**Owner decision 2026-11-18:** the project is experimental, and a single-language tree
(one type system across the host/device wall, kernels under jscpd and CodeScene for the
first time, the ABI wall's exemptions shrinking) is worth a nightly toolchain. This doc
is the plan and becomes the record: each phase appends what it measured, in place, dated.

**Treat it as a hypothesis.** Every phase has a stop condition. A well-evidenced reason to
stop (a kernel that cannot be made bit-identical, a decode regression that cannot be
closed) is a successful outcome and is written here like any other result.

## 1. What the spike established (2026-11-18)

Spike tree: `/var/cache/rivoli/scratch/rust-amdgpu-spike/` (scratch, not vendored).

- Toolchain: `rustc 1.102.0-nightly (59cbdee2e 2026-11-17)`, target `amdgcn-amd-amdhsa`,
  `-Ctarget-cpu=gfx1151`, `-Zbuild-std=core`, features `abi_gpu_kernel`,
  `core_intrinsics`, `gpu_intrinsics`, `stdarch_amdgpu` (+ `asm_experimental_arch` for
  barriers). rustc's LLVM is 23.1.5; hipcc's is ROCm's 23.0.0git fork.
- Building blocks exist in core: `core::arch::amdgpu` (workitem/workgroup ids,
  `s_barrier`, `ds_bpermute`, `readlane`, wave reductions),
  `core::intrinsics::gpu::gpu_launch_sized_workgroup_mem` (dynamic LDS),
  `core::intrinsics::gpu::amdgpu_dispatch_ptr`, and `core::intrinsics::{exp, sqrt,
  fmuladd}` which lower to the same LLVM intrinsics hipcc's device libs use.
- **Bit-identity, witnessed:** `swiglu` and `rmsnorm_single` from `activation.hip`, loaded
  through `hipModuleLoad` beside a hipcc `--offload-device-only` build of the same file,
  768 cells over 64 seeds (incl. exp overflow/underflow, −0, subnormals, ragged tails):
  0 mismatches. Witness CLEAN on every arm (0 foreign KFD, GTT 17 MiB pre-arm).
- **Red proof, and its first failure:** a no-`fmuladd` plant was bit-identical at 8 seeds
  — a 1-ulp change in the sum is usually erased by the `sqrt` (CPU model: the sum moves in
  13/80 draws, `inv` in 5/80). At 64 seeds it reddened 3/256 draws (seeds 17, 19, 57;
  ~95% of outputs when it fires). A multiply-form `swiglu` plant
  (`x·(1/(1+e^-x))` vs the division form) reddened every cell (~20% of outputs) — the
  form hazard `activation.hip` documents as undemonstrated, now demonstrated in fp32.

### Hazards every port must handle

1. **FMA contraction.** hipcc compiles with `-ffp-contract=fast-honor-pragmas`; every
   `fmul`+`fadd` it emits carries `contract` and fuses. Rust never contracts. Bit-identity
   requires an explicit `fmuladd` at exactly the sites hipcc fused — read hipcc's IR
   (`--cuda-device-only -S -emit-llvm`) or ISA, never guess. A missing one passes most
   seeds: that is why the bit-identity gate needs ≥ 64 seeds and adversarial inputs.
2. **No workgroup-scope fence in Rust.** `core::sync::atomic::fence` is system scope and
   adds a `buffer_gl1_inv` per barrier (~5% on `rmsnorm_single`). HIP's `__syncthreads`
   on gfx11 is `s_waitcnt lgkmcnt(0); s_barrier; buffer_gl0_inv`; the spike's `asm!`
   barrier matches it exactly and timed at parity. Atomics have the same scope question:
   HIP `atomicAdd` scope must be matched, not assumed.
3. **Small-launch gap, UNEXPLAINED.** Rust `swiglu` is 1.38–1.68× hipcc at 6k–12k
   elements (2.0 → 3.3 µs), 1.01–1.06× at 1M, survives order reversal. The only kernel-
   descriptor difference found: Rust enables the dispatch-ptr user SGPRs (blockDim read
   from the packet; HIP reads implicitarg). `rmsnorm_single` uses the same read with no
   gap, so that is not yet the cause. **Decode is launch-bound; this gates the migration.**

## 2. Target architecture

- A device crate (working name `crates/kernels`) — `no_std`, nightly, built for
  `amdgcn-amd-amdhsa` by `crates/backend/build.rs` under the `rocm` feature, with its own
  target dir under `OUT_DIR`, its toolchain pinned in-tree, the code object embedded with
  `include_bytes!` and loaded once through `hipModuleLoadData`.
- The `rivoli_*` C++ launchers (argument guards, grid geometry, return-code convention)
  move to Rust host code; `launchers!` keeps its one-declaration-two-halves shape but its
  second half becomes a module-function launch instead of an `extern "C"` call. The
  guards' positive codes and the `-(hipError_t)` convention move with them unchanged.
- Host-only C++ (`async.hip`, `rivoli_device_sync`, the memcpy/fill entry points) becomes
  direct FFI to `libamdhip64`; no C++ remains.
- The `repr(C)` types that cross the wall (`abi.rs`, `ExpertDesc` …) move to a `no_std`
  crate both sides compile, so host and device share ONE definition — the unification
  payoff. Where that is not possible, a compile-time layout assert on both sides.
- `hipcc` leaves the build. It remains only in the bit-identity reference (§3).

## 3. Gates (all of them must be able to go red)

- **Bit-identity reference.** Tag `archive/hip-kernels` at the migration base. An
  on-demand GPU harness builds the tagged `.hip` sources with hipcc
  (`--offload-device-only --no-gpu-bundle-output`) into scratch and compares every
  migrated kernel against its Rust twin: ≥ 64 seeds, adversarial inputs (exp/overflow
  edges, −0, subnormals, NaN where the kernel's contract admits it, ragged tails), every
  shape class the engine launches. Byte-identical or RED. Each lane ships a recorded red
  proof with a plant shown adversarial on CPU first (§1's lesson: the first plant was
  sub-threshold).
- **The existing oracle suites** (`crates/*/tests/kernel_*.rs` etc.) stay green
  unmodified, under the GPU lease.
- **Kernel census** (`kernel_coverage.rs`) and the liveness census adapt to the new
  declaration form; both ends stay checked.
- **jscpd, CodeScene, line caps now cover kernels.** Expect findings the `.hip` tree never
  had to face; they are addressed, not exempted, unless argued in place.
- **Timing, per kernel class.** A byte-identity gate cannot see a slowdown. Each lane
  reports per-kernel µs vs hipcc at the engine's real shapes; the end of the migration is
  a decode tok/s A/B (GLM) and `tests/parity-glm.sh`.

## 4. Phases

- **Phase 0 (parallel):**
  - **0a infra** — device crate, build pipeline, loader, `launchers!` rewrite, host FFI for
    `async.hip`/sync/memcpy/fill, the bit-identity harness with its red proof, census
    adaptation; migrates `activation.hip` and `residual.hip` end to end as the proof.
  - **0b prelude** — `common.hpp` and its eight headers (`formats`, `reduce`, `rowview`,
    `actquant`, `dot_fp8`, `dot_i4`, `dot_vq`, `dot_f4`; ~1300 lines) as Rust device
    modules. Every kernel includes `common.hpp`, so nothing in Phase 1 can start before it.
  - **0c launch gap** — explain the §1.3 gap or bound it; fix or a measured verdict. If it
    cannot be closed, the migration's cost is decode latency and the owner decides.
- **Phase 1 (four lanes, after 0a+0b merge):** L1 `linalg`, `vmm`, `headtail`, `fwd` ·
  L2 `moe`, `moe_f4`, `recurrent` · L3 `mla`, `kvcompress`, `blockindex`, `indexer` ·
  L4 `attn`. Each lane deletes its `.hip` files on landing.
- **Phase 2:** hipcc out of `build.rs`, `kernels/` directory gone, docs and
  `kernel-inventory.md` rewritten, decode A/B and parity gate, CLAUDE.md's jscpd exemption
  count re-derived.

## 5. Record

(Phases append here, dated, with witnessed numbers.)
