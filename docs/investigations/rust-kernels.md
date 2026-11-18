---
status: live
scope: engine
verdict: OPEN 2026-11-18 — migrate every HIP C++ kernel (crates/backend/kernels, 15 .hip + 8 .hpp, 8115 lines, 66 launchers) to Rust on amdgcn-amd-amdhsa, owner decision for a smaller unified codebase. The spike proved bit-identity on two kernels and named three hazards (FMA contraction, missing workgroup-scope fence, a ~1 µs small-launch gap, isolated by Phase 0c 2026-11-18 to the dispatch-packet blockDim read); this plan is a hypothesis, and a measured reason to stop is a valid outcome.
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
   > **CORRECTED 2026-11-18 (Phase 0a):** use `fma`, not `fmuladd`, at a fused site.
   > `llvm.fmuladd` leaves the fuse-or-not choice to the backend's profitability check, and
   > that choice IS the bits; `llvm.fma` always fuses, which is what the `v_fma`/`v_fmac` in
   > hipcc's ISA is. Also: this hipcc emits `contract` FLAGS rather than `fmuladd` calls
   > (`fmul contract`/`fadd contract` in the IR) and fuses in the backend, so the IR shows
   > candidates and only the ISA shows the decision.
2. **No workgroup-scope fence in Rust.** `core::sync::atomic::fence` is system scope and
   adds a `buffer_gl1_inv` per barrier (~5% on `rmsnorm_single`). HIP's `__syncthreads`
   on gfx11 is `s_waitcnt lgkmcnt(0); s_barrier; buffer_gl0_inv`; the spike's `asm!`
   barrier matches it exactly and timed at parity. Atomics have the same scope question:
   HIP `atomicAdd` scope must be matched, not assumed.
   > **CORRECTED 2026-11-18 (Phase 0a): the spike's barrier was UNDER-fenced.** It copied
   > `rmsnorm_single`'s ISA, where hipcc's waitcnt pass had elided the global-memory waits
   > because nothing was in flight. `rope_interleave`'s ISA shows the full release:
   > `s_waitcnt vmcnt(0)` before `s_barrier`, because each thread's global LOADS must land
   > before any thread's in-place STORES (a cross-lane write-after-read). Opaque asm cannot
   > be elided, so `crates/kernels/src/gpu.rs::sync_threads` waits on all three counters
   > (`vmcnt`, `lgkmcnt`, `vscnt`), then the `llvm.amdgcn.s.barrier` INTRINSIC (convergent —
   > Rust inline asm is not, so an asm barrier could be duplicated into divergent branches),
   > then `buffer_gl0_inv`; the two asm blocks are what order memory around the intrinsic,
   > which is `IntrNoMem`. A wait on a zero counter costs a cycle. And a fourth hazard the
   > spike did not meet: **`f32::max`/`min` are not `fmaxf`/`fminf`** — core lowers them to
   > `llvm.maximumnum`/`minimumnum`, HIP to `llvm.maxnum`/`minnum` (flagged by 0b, checked
   > in core's source); the device crate binds the LLVM intrinsics directly.
3. **Small-launch gap, UNEXPLAINED.** Rust `swiglu` is 1.38–1.68× hipcc at 6k–12k
   elements (2.0 → 3.3 µs), 1.01–1.06× at 1M, survives order reversal. The only kernel-
   descriptor difference found: Rust enables the dispatch-ptr user SGPRs (blockDim read
   from the packet; HIP reads implicitarg). `rmsnorm_single` uses the same read with no
   gap, so that is not yet the cause. **Decode is launch-bound; this gates the migration.**
   > **CORRECTED 2026-11-18 (Phase 0c, §5): UNEXPLAINED → ISOLATED.** The cause IS the
   > dispatch-packet blockDim read: a hipcc kernel reading blockDim the same way pays the
   > identical ~1.1–1.2 µs at 24–48 workgroups, and Rust kernels taking blockDim from a
   > constant or an explicit kernarg match hipcc within 0.1 µs. `rmsnorm_single` showed no gap
   > because it launches ONE workgroup, where the penalty is ~0 — the sentence above drew its
   > exoneration from the one shape that cannot show it. Rule: never read blockDim from the
   > dispatch packet (§5, Phase 0c).

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

> **AMENDED 2026-11-18 (Phase 0a), three points the build proved:**
> - **ROCm's device libraries stay a build input.** `tanhf`, `pow`, `sin`, `cos` are OCML
>   (`__ocml_*`), which hipcc inlines from `ocml.bc`; Rust has no equivalent that is
>   bit-identical. Rustc can consume the SAME bitcode: `-Clinker-plugin-lto` plus the
>   `.bc` files on the link line (exactly hipcc's `oclc_*` set, read off `hipcc -###`) makes
>   lld LTO them into the Rust kernels, inlined (probe: a `__ocml_tanh_f32`/`pow`/`cos`
>   kernel linked with no calls left in the ISA). hipcc leaves; `/usr/lib/amdgcn/bitcode`
>   does not. One lld wrinkle: rustc's version script names each `.kd`, which under
>   linker-plugin LTO does not exist until LTO runs — `--undefined-version` is required.
> - **The shared ABI crate carries the kernel PARAMETER LISTS, not just the structs.**
>   Every Rust kernel takes one `rivoli_abi::kernarg` struct by value (metadata: one
>   `by_value` arg of `size_of`; ISA `s_load` offsets are the `repr(C)` offsets — checked),
>   so a kernel's parameters are spelled once instead of three times (signature, launch
>   call, extern mirror). This was forced as much as chosen: jscpd, scanning kernels for the
>   first time, found 26 clones on the first build, nearly all parameter lists repeated
>   between Rust launchers and Rust kernels. Launch geometry constants live beside them
>   (`rivoli_abi::geom`), so a launcher and its kernel name ONE block size — Phase 0c's tie.
> - **The C launcher convention survives as Rust.** `launchers!` rows keep one form per
>   launcher: `launch_x -> rivoli_x` (hipcc, `extern "C"`) or `launch_x => rivoli_x(KernArg
>   {..}, ..)` (Rust); migrating a row is that arrow plus its call expression, and the
>   wrapper every caller sees is unchanged.

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

### Phase 0c — small-launch gap (2026-11-18)

**Verdict: CAUSE ISOLATED, FIX DEMONSTRATED AT PARITY.** Full report (scratch):
`/var/cache/rivoli/scratch/rk-launchgap/REPORT.md`; verified against raw output by the
coordinator.

- **Cause:** reading blockDim from the AQL dispatch packet (`gpu::amdgpu_dispatch_ptr()+4`).
  A hipcc kernel doing the same (`H:swiglu_dp`) pays the identical penalty; Rust kernels
  taking blockDim from a constant (`R:swiglu_256`) or an explicit kernarg (`R:swiglu_karg`)
  match hipcc within 0.1 µs. Compiler, code-object metadata and kernarg size (288 vs 32 B)
  refuted as causes.
- **ev1** (single launch between events, median of 301), n = 6144 / 24 WGs, pass 0, µs:

  | arm | µs |
  |---|---|
  | H:swiglu | 5.01 |
  | H:swiglu_256 | 5.05 |
  | R:swiglu_256 | 5.01 |
  | R:swiglu_karg | 5.09 |
  | H:swiglu_dp | 6.13 |
  | R:swiglu (packet read) | 6.21 |
  | nop | 4.65 / 4.69 |

  Penalty vs workgroup count: ~0 (1 WG), ~0.17 (4), ~1.2 (24), ~1.15 (48), ~1.0 (256),
  ~0.8 (1024). The 1M row is inconclusive (2 µs pass-to-pass swing).
- **Arms:** RED (blockDim 128 planted into `R:swiglu_256`, check-only) rc=1, 6/7 cells
  mismatched, only that arm (n = 256 is one WG and cannot differ). M rc=0, BIT-IDENTICAL,
  8 arms × 7 sizes; nop arms leave the poisoned buffer untouched. **Witness: both empty,
  GTT 17 MiB pre-arm.** Host loadavg 33.5/32 cores during arms (sibling builds) — noise in
  the throughput columns, recorded, not a discard.
- **HIP's route is not expressible in nightly 1.102:** rustc rejects
  `llvm.amdgcn.implicitarg.ptr` (`expected ptr addrspace(4) (), found ptr ()`); declaring
  it `extern "C"` ICEs.
- **Mechanism: UNMEASURED.** Hypothesis: the packet lives in the host-coherent AQL ring, so
  its scalar loads are likely uncached and contended across concurrent WGs, on every wave's
  critical path.

**RULE for the device crate: never read blockDim from the dispatch packet.** A per-kernel
constant where the launcher fixes the block size; an explicit argument where it is chosen
at run time (rope's `seg/2`, append_kv's `kvl`, gated_delta's `head_dim`,
rmsnorm_gate_heads/gdn/index_* blocks). **Every constant carries a launcher-side check tying
the launch geometry to it** — a mismatch is silent wrong output (0c's RED arm). Fixed sizes
that are not 256: 128 `act_quant_f4_rotated`, 1024 `hc_pre`, 32 `hc_head_collapse_gate`,
`GQA_THREADS`, `GATHER_ATTN_THREADS`, `HB*SUBW`.

### Phase 0a — infra, deviceless half (2026-11-18)

**Landed (branch `wave/rk-infra`):** `crates/kernels` (device crate, pinned
`nightly-2026-11-18` = `rustc 1.102.0-nightly (59cbdee2e)`, not a workspace member) built by
`crates/backend/build/device.rs` under `rocm` only, into `$OUT_DIR/kernels-target`, embedded
and loaded once (`src/module.rs`, `hipModuleLoadData`); `crates/abi` (`rivoli-abi`, `no_std`,
compiled by both sides: kernel argument blocks, launch geometry, `CompGeom`/`CompFinish`);
`activation.hip`, `residual.hip`, `async.hip` deleted — 13 kernels plus `touch_region` are
Rust, their launchers' guards/geometry/codes are `src/rk_launchers.rs`, and async.hip's host
half, `rivoli_device_sync`, the two device-to-device copies and `fill_u32` are direct FFI
(`src/hiprt.rs`; `fill_u32` is `hipMemsetD32Async` on the null stream). One C-ABI export
bridges the migration: `rivoli_act_quant_f8`, which `moe_f4.hip` still calls C-to-C.

**Deviceless gates, measured:**

- Workspace builds both arms; clippy `--all-targets` clean on both arms; jscpd clean (it
  found **26 clones** on the first kernel-covering build — the plan predicted findings —
  all removed by restructuring, none exempted: the kernarg structs (§2 amendment), a shared
  `tree` ladder, helper extraction in the gate's rows).
- CodeScene: every new or touched file 10.0. `rk_launchers.rs` first scored 9.68 (Primitive
  Obsession on the C launchers' `int` vocabulary); a `Rows` grid-extent newtype — the unit the
  review named, and CLAUDE.md's newtype rule — took it to 10.0, so it carries no exemption.
  The whole-tree CodeScene test is RED on the base commit itself: 31 files score below 10, all
  of them red at the base (one, `kernel_qwen_harness.rs`, carries a one-comment edit here and
  scored 9.68 before it) (e.g. `v4oracle/compress.rs` 6.06, `hip_attn.rs` 9.68 — files whose
  bytes are the base's), so that test cannot be green on any branch until they are triaged.
- `CARGO_TARGET_DIR=/var/cache/rivoli/target/rk-infra cargo test --workspace
  --no-default-features --no-fail-fast`: 500 passed, 1 failed — the CodeScene test above,
  and nothing else.
- **The deviceless arm needs neither nightly nor ROCm — shown, not argued:**
  `env RUSTUP_HOME=<a home holding ONLY the stable toolchain> HIPCC=/bin/false
  RIVOLI_DEVICE_LIBS=/nonexistent CARGO_TARGET_DIR=/var/cache/rivoli/target/rk-infra-deviceless
  cargo test --workspace --no-default-features --no-fail-fast` into a freshly deleted target
  dir: 500 passed, 1 failed (the same CodeScene test). Red proof that the environment really
  lacks the device toolchain: the same environment (with the real hipcc) building
  `-p rivoli-backend` fails in `build/device.rs` on `toolchain 'nightly-2026-11-18…' is not
  installed`.
- Census: three censuses green (66/66/0; 53 shipped + 13 classified; 14 Rust kernels, one
  host handle each), three plants red in `gate-red-proofs.md` §18, plus the unplanted finding
  that the UNADAPTED oracle census passed with 13 launchers missing.
- ISA census (fusion opcodes per kernel, hipcc vs Rust): identical for `swiglu`,
  `swiglu_clamped_bf16`, `situ_glu_f32`; `rope_*` differ only in an equivalent encoding of
  the same fused op (`v_fma` with a negated operand vs `v_fmac` over a negated product); the
  reductions differ in unrolling, which the counts cannot see through — the device gate is
  the judge. Correctness review re-derived every fused site from the IR's `contract` pairs
  (left product fused, as 0b also found) and the code-object descriptors (denorm mode, IEEE,
  WGP, wave32) — all match.
- The gate's classifier (`tests/bit-identity.sh`) was found by review UNABLE TO GO GREEN:
  its zero-examined regex matched the mismatch clause of every healthy line. Rewritten on
  explicit `BIT-IDENTICAL`/`RED`/`VACUOUS` tokens, a row count checked against the binary's
  `RESULT` total, and an exact-row `--expect-red=ROWS`; eight stub-stdout arms red-proof it
  (§18). Per the owner's decision of 2026-11-18 the compare treats **NaN as a class** (both
  sides NaN = equal, everything else bit-exact) and prints `nan-payload-only` per row; the
  output poisons are therefore two distinct FINITE sentinels, since two NaN poisons would
  now compare equal and hide an element neither twin wrote.
- Known gaps, stated: clippy does not run on `crates/kernels` (the pinned toolchain is
  `minimal`; its manifest denies rustc warnings); `fill_u32` as `hipMemsetD32Async` into a
  `hipMemMap`'d pool slot is unverified on this ROCm; `attn_res`'s private `p[16]` lands in
  scratch (144 B) where hipcc keeps it in registers — a speed question the TIME line answers;
  0b's probe families (`/var/cache/rivoli/scratch/rk-prelude/probe/`) are not yet rows.
- **Barrier, versus Phase 0b's:** both wait `vmcnt`/`lgkmcnt`/`vscnt` and invalidate L0;
  0b's spells `s_barrier` inside the asm, this one uses the `llvm.amdgcn.s.barrier` intrinsic
  between two asm blocks, because the intrinsic is `convergent` and inline asm is not (LLVM
  may duplicate a non-convergent barrier into divergent branches). One must win at merge.
- Bit-identity gate built; its CPU plant model green (rope unfused: 23212/131072 outputs,
  64/64 seeds). Device arms: pending the GPU lease.
