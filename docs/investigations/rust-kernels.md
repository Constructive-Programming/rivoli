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

### 5.1 Phase 0b — the prelude (2026-11-18)

**What landed.** `common.hpp` and its eight headers became `crates/kernels/src/prelude/`
(10 modules). 0b's minimal skeleton is `crates/kernels/Cargo.toml` + `src/lib.rs`, which 0a
owns on merge. Templates became const generics (`R` rows, `U` unroll depth) plus one trait.
Three structural changes, each scored by the device gate below:
- `dot_i4.hpp` and `dot_f4.hpp` are ONE loop: `dot_nib::dot_nib_wave_r` over a `NibFormat`
  (decode, scale type, group width). Both dword passes compute `s·(x0·n0+…+x3·n3)`; both
  tails compute `x·n·s` left to right (fp4 only hoisted the scale LOAD).
- The single-row/batched C++ twins (`fp8_dot_strided` vs `_r`, `dot_i4_wave` vs
  `dot_i4_wave_r`) are one function each, the C++ unroll depth kept as a const parameter.
  Each C++ note already claimed R=1 bit-identical to its single-row form.
- `block_sum_lds(v, red, tid, block_dim)` takes the block size from the kernel (0c's rule:
  no helper reads the dispatch packet). The host-only `next_pow2` was not ported: a device
  crate cannot serve a host launcher, and `u32::next_power_of_two` covers it.

**Toolchain answers (§1 hazards, read from the IR/ISA, not guessed):**
- *FMA.* hipcc fuses `a·b + c·d` as `fma(a, b, c·d)` (the LEFT product), then each `+ e·f`
  as `fma(e, f, ·)`; the shared `rowview::quad` is this shape. `acc += s·q` is
  `fma(s, q, acc)`; `acc += x·l·s` is `fma(x·l, s, acc)`. `e4m3f` fuses `1 + m·0.125`
  (exact, reproduced anyway). Nothing else in `formats`, `actquant`, `reduce` or
  `activation` has a fused site; ocml's tanh brings its own explicit `fmuladd`s.
- *Intrinsics.* Some `core` functions lower to lookalike LLVM intrinsics, so
  `prelude::llvm` declares the exact ones; its header lists each case. **`situ_glu`'s tanh
  is transliterated from ocml IR, so it is pinned to this ROCm device-lib build**; a ROCm
  upgrade can move the C++ side.
- *Barrier.* `asm!` with `s_waitcnt vmcnt(0) lgkmcnt(0)`, `s_waitcnt_vscnt null, 0x0`,
  `s_barrier`, `buffer_gl0_inv`. That is hipcc's WGP-mode workgroup release: a hipcc probe
  with a global store before `__syncthreads` emits the `vscnt` wait. hipcc drops each wait
  when its counter is provably zero; the asm always emits them (free when nothing is
  outstanding). The first spelling had only `lgkmcnt(0)`. It matched every probe kernel,
  because none stores to global before a barrier, but it was unsound for kernels that do —
  caught by review, not by a gate. `block_sum_lds` keeps 2 + log2(bd) barriers (6 in the
  two-reduction probe, both sides).
- *No unroll pragma in Rust.* `rowview::for_strided::<U>` issues U body copies as
  straight-line code, then a scalar remainder. An inner `for u in 0..U` was measured NOT to
  unroll fully: R=2 int4 got 8 activation loads where hipcc has 20.
- *Not expressible: `gu8p` (address_space(1)).* Bits are unaffected. In a descriptor-loop +
  atomic kernel (moe_f4's shape), Rust emits `flat_load` for the fp4 weight dword and scale
  byte: 5 `flat_load_b32` + 7 `flat_load_u8`, where hipcc+gu8p emits `global_load`.
  Without the atomic, Rust infers global unaided. Two workaround attempts:
  - `assert_unchecked(!is_shared(p))` + `!is_private(p)` is GONE from the optimized IR
    (same census), as `__builtin_assume` was in C++ (2026-08-08).
  - `llvm.amdgcn.raw.buffer.load.i32` compiles (repr_simd + simd_ffi) to `buffer_load_b32`
    (vmcnt-only), but it is a different instruction.

  **L2 (`moe_f4`) OBLIGATION:** a flat-vs-global load census of
  `moe_gateup_f4`/`moe_down_f4` plus a timing row against hipcc. A buffer_load fallback
  ships only with a measured timing at parity.
- LTO merges byte-identical gpu-kernel functions and drops the second's `.kd`, so the link
  fails (`k_vq_wave_r1` ≡ `k_vq_wave`).

**Deviceless gate — per-kernel ISA census** (`census.py`):
- *What it compares.* Over every kernel label in both code objects (36; a missing kernel
  is reported, not skipped), the exact per-kernel count of every FP, cross-lane, barrier
  and load mnemonic. VGPRs compare by occupancy (24-register granules from 1536).
- *Result.* **36/36 op counts identical** to hipcc. **35/36 at equal occupancy; the
  exception is `k_i4_wave_r2`: VGPR 96 → 103, i.e. 16 → 12 waves/SIMD** (hipcc sits
  exactly at the granule edge). Probe kernels are not the engine's kernels, so this is
  **an L2 obligation:** the real `moe_*_i4` R=2 kernels get a VGPR/occupancy census and a
  timing row.
- *Red proof.* The census goes DIFF under the formats, common, reduce, barrier
  (`s_barrier` 6 → 4), fp8 and nib(R=2) plants.
- *Blind spots.* It does not see the actquant (select vs minnum), situ (branch threshold),
  vq (operand order) or nib R=1 plants. Same counts, different values — which is why the
  device sweep is the gate and the census only a pre-check.

**Device gate — design.** All files are in `/var/cache/rivoli/scratch/rk-prelude/probe/`:
- `probe.hip` is built from the tag's headers; the Rust twins are in `rk/`.
- `harness.cpp` compares bytes on the device and fails setup if a group examines 0 cells.
- `build-all.sh` is deviceless and writes `artifacts.sha256`.
- `run-arms.sh` + `check-arm.py` judge each red arm by WHERE it is red.

*These gates live in scratch, so for anyone else they cannot go red; they move into the
tree with 0a's harness.*

What the device gate runs:
- EXHAUSTIVE over every small domain:
  - all 2^32 f32 for `rbf16`, `f2bf16`, `f2e4m3_rne`, `f2e2m1`, `moe_fixed`, `e2m1f`,
    `nib` (k = 0..7), `blk_shift`, `absorb_nquad`;
  - `fast_round_scale` at 3 `max_inv`;
  - both round trips at 10 scales (the power-of-two domain, 0.3, subnormal, ±0, inf, NaN,
    −2);
  - `swiglu_clamped` over the whole bf16×bf16 grid (its true domain) at 4 limits;
  - `situ_glu` with each operand swept over 2^32 against 11 partners × 2 beta pairs;
  - all codes of `bf16f`, `e4m3f`, `e8m0f`, and the LUT.
- Seeded, 64 seeds per cell, three classes:
  - normal draws with fixed −0/subnormal entries;
  - 2^±60 magnitudes;
  - nasty: inf, NaN payloads incl. sNaN, −0, subnormal, NaN fp8 codes, e8m0 0/0xff,
    fp16 inf/NaN.

  Every lane's value is compared, not only lane 0. The seeded cells:
  - `wave_sum`/`wave_max`;
  - `block_sum_lds` twice back to back at bd = 32…1024;
  - fp8 wave/split-K × R ∈ {1,2} over 10 (i_dim, block) shapes;
  - int4 over 10 dims and fp4 over 10, including dim 1028, whose odd rows are misaligned
    (tail path) next to aligned ones, plus multi-trip and odd-trip counts;
  - VQ over 5 dims incl. partial groups.

**Red proofs, CPU first** (`/var/cache/rivoli/scratch/rk-prelude/cpu/`, output
`redproof-final.txt`). Codec plants run against the real Rust source; the dot and tanh
plants against scalar models. Each plant must disagree where its defect applies and agree
where it does not:

| family | plant | CPU: disagrees | CPU: agrees |
|---|---|---|---|
| formats | subnormal tie rounds half-up | exactly the 6 inputs (2.5, 4.5, 6.5)·2^-9, ± | the other 2^32 − 6 |
| actquant | clamp via minnum/maxnum | every input with NaN `x/s` (16 777 214 per finite scale, 2^24 at 0/±inf) | every non-NaN quotient (0 elsewhere, all 10 scales); at s = NaN never — the final `·s` re-poisons both |
| common | swiglu gate clamped both sides | 1 063 093 058 of the 1 063 124 992 bf16×bf16 cells with `!(g ≥ −10)` (NaN gates included) | 0 outside that set |
| situ | tanh crossover 0.625 → 0.5 | 1 723 918 of the 4 194 304 inputs with \|x\| in [0.5, 0.625) | 0 outside that set |
| reduce | wave_sum as xor butterfly | lanes 1..31 in 4096/4096 waves | lane 0 in 4096/4096 |
| fp8 | acc `+=` instead of fma | 144/192 random rows | 0/192 exact-arithmetic rows |
| nib | tail `x·(n·s)` | 33–35/64 rows at dims with a tail | 0/64 at 2048, 6144 (no tail) |
| vq | right product fused | 128/192 random rows | 0/192 exact rows |
| barrier | no trailing barrier | census `s_barrier` 6 → 4 | — (a race; bits may not show it) |

The first formats plant considered (`a <= 2^-10` → `<`) is a NO-OP. That tie falls into
the subnormal path and rounds to the same even zero (CPU: 0 inputs differ). It is the
spike's sub-threshold-plant lesson again, caught on the CPU this time instead of on the
GPU.

**Device results: PENDING the GPU lease** (READY-FOR-GPU posted 2026-11-18).
