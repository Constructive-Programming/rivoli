---
status: live
scope: engine
verdict: OPEN 2026-11-18 — migrate every HIP C++ kernel (crates/backend/kernels, 15 .hip + 8 .hpp, 8115 lines, 66 launchers) to Rust on amdgcn-amd-amdhsa, owner decision for a smaller unified codebase. The spike proved bit-identity on two kernels (and Phase 0a/0b proved the whole prelude bit-identical on device, 2026-11-18: 222/222 probe groups, eight plants red where their CPU proofs say) and named three hazards (FMA contraction, missing workgroup-scope fence, a ~1 µs small-launch gap, isolated by Phase 0c 2026-11-18 to the dispatch-packet blockDim read); this plan is a hypothesis, and a measured reason to stop is a valid outcome.
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
   > **2026-11-18 (0a/0b merge):** that `sync_threads` now lives in `prelude::reduce`, and
   > the `maxnum`/`minnum` bindings in `prelude::llvm`; `gpu.rs`'s copies are gone.
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
>
> > **2026-11-18 (0a/0b merge):** `tanhf` now comes from the prelude's transliteration of
> > ocml's body (`prelude::activation`, 0b), measured identical on the device by 0b's sweep;
> > `pow`/`sin`/`cos` stay LTO-linked. Both routes pin the build to this ROCm's device libs.

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
  `hipMemMap`'d pool slot is unverified on this ROCm (CLOSED 2026-11-18, §5's merge-review
  lease, arm T); `attn_res`'s private `p[16]` lands in
  scratch (144 B) where hipcc keeps it in registers — a speed question the TIME line answers;
  0b's probe families (`/var/cache/rivoli/scratch/rk-prelude/probe/`) are not yet rows.
- **Barrier, versus Phase 0b's:** both wait `vmcnt`/`lgkmcnt`/`vscnt` and invalidate L0;
  0b's spells `s_barrier` inside the asm, this one uses the `llvm.amdgcn.s.barrier` intrinsic
  between two asm blocks, because the intrinsic is `convergent` and inline asm is not (LLVM
  may duplicate a non-convergent barrier into divergent branches). One must win at merge.
  > **RESOLVED 2026-11-18 at the 0a/0b merge (coordinator's ruling): the intrinsic form won**
  > and now lives in `prelude::reduce::sync_threads` (dated note there); `gpu.rs`'s copy is
  > deleted, as are its shuffles and `fmaxf`/`fminf` — the prelude's are the only spelling.
  > 0b's ISA census re-run against the merged prelude (their probe crate pointed at this
  > tree, their `census.py`, the tag's `hip.dis`): 36 kernels, the same single DIFF they
  > recorded (`k_i4_wave_r2` occupancy), `s_barrier` 12/12 — the barrier change moved no
  > counted instruction. This branch's own kernels after the merge: `swiglu`,
  > `swiglu_clamped_bf16`, `situ_glu_f32` still count identical to hipcc (the last now through
  > the prelude's ocml-`tanh` transliteration instead of the LTO-linked body); `attn_res`'s
  > counts moved (`v_add` 7 → 21, `v_dual_add` 8 → 1 — VOPD pairing, which this census cannot
  > see through), so for it and the other reductions the device gate is the only evidence.
- Bit-identity gate built; its CPU plant model green (rope unfused: 23212/131072 outputs,
  64/64 seeds). Device arms: pending the GPU lease. (Run 2026-11-18 — leases 1 and 2 below.)

### Phase 0a — device arms, lease 1 (2026-11-18, commit `a2abd64`)

Three arms, rh-anine, each through `run_arm`: **witness CLEAN on all three** (empty KFD
witness file, GTT 17 MiB pre-arm; host loadavg 10.5→16.0 on A, 15.3→14.7 on R, 10.6→3.1
on W — sibling builds, recorded, not a discard). Logs:
`/var/cache/rivoli/scratch/bit-identity/20261118T184545-4013011` (A),
`…/20261118T184629-4015490` (R), `/var/cache/rivoli/scratch/rk-infra/gpu/W.{out,err}` (W).

**Arm A (real binary `bit_identity-efca1da5680a0280`): 12 of 13 rows BIT-IDENTICAL; `hc_post`
RED** — 26 words over 23 of 192 cells, NaN-payload-only 0 on every row. Cause, read off hipcc's ISA afterwards: `post·x + comb₀·res₀` is `a·b + c·d`, and hipcc
fuses the LEFT product (`fma(post, x, comb₀·res₀)`: one `v_mul`, four `v_fmac`) — the rule 0b
recorded; the port fused the right one. A CPU check could not have told them apart; the
gate did. The script's verdict line ALSO read RED for a second reason: libtest prints
`test <name> ... ` with no newline before a `--nocapture` body, so the first row's token
(`BIT-IDENTICAL swiglu`) landed mid-line and the anchored classifier counted 12 rows of 13
and refused — the fail-safe direction, but it would refuse a clean arm too.

Per-kernel µs, A's TIME lines (min of three 200-launch batches, hot shape, null stream):

| kernel (shape) | hipcc µs | Rust µs | ratio |
|---|---|---|---|
| swiglu (6144) | 2.05 | 2.04 | 0.996 |
| swiglu_clamped_bf16 (2048) | 2.05 | 2.02 | 0.985 |
| situ_glu_f32 (2048) | 2.15 | 2.12 | 0.988 |
| rmsnorm_single (6144) | 7.59 | 7.53 | 0.992 |
| rmsnorm_centered_single (4096) | 6.14 | 5.72 | 0.931 |
| rmsnorm_weightless_batch (32×128) | 2.99 | 2.55 | 0.853 |
| act_quant_f8 (1×7168) | 3.26 | 3.28 | 1.008 |
| rope_interleave (32,128,128) | 4.78 | 4.87 | 1.018 |
| rope_split_half (32,128,128) | 4.78 | 4.86 | 1.017 |
| **hc_pre (1×4096)** | **14.41** | **55.58** | **3.856** |
| hc_post (1×4096) | 2.57 | 2.47 | 0.962 |
| **attn_res (1,9,7168,9)** | **53.07** | **73.40** | **1.383** |
| gated_residual_collapse_f32 (4×2560) | 2.75 | 2.36 | 0.860 |

The small-launch gap is gone (Phase 0c's rule holds: ≤ 1.02 on every elementwise kernel).
Two real slowdowns: `hc_pre` 3.9x — its C carries `#pragma unroll 8` on both streaming loops
and records that load depth WAS the cost; the port had none — and `attn_res` 1.38x, whose
private `p[16]` sat in 144 B of scratch where hipcc keeps it in registers.

**Arm R (planted `rope` lo unfused): the plant reddened exactly its two rows** —
`rope_interleave` 275/320 cells (53256 words), `rope_split_half` 275/320 cells (53002 words) —
with every other row unchanged from A (including A's `hc_post` red). The script's
`--expect-red` verdict was nevertheless NOT met, for A's two reasons (a third red row, and the
merged first line), so the red proof is recorded as OBSERVED and its scripted verdict is owed
to the re-run.

**Arm W (the whole device battery, `--test-threads=1`): 705 passed, 1 failed, 1 ignored.**
The pre-arm freshness check was `cargo build --workspace --tests` (0 `Compiling`, 118
`Fresh`; the guard refuses `cargo test --no-run` without the lock) — and it is NOT an exact
proxy: inside the lock `cargo test` still printed `Compiling rivoli-engine` and `Finished … in
0.41s` (W.err line 5), cause not determined. A sub-second hold, but "nothing compiled in the
lock" is false for this arm; the next battery needs a proxy that matches `cargo test`'s
units. The one failure is the CodeScene test that is red on the base commit; the ignored test
is the bit-identity gate. Every oracle suite over the 13 migrated launchers, the
stream/event/timeline FFI and the fetch path passed unmodified.

> **CAUSE FOUND 2026-11-18 (after lease 2):** `cargo test` also builds every package's
> EXAMPLE targets, and `cargo build --tests` does not — `crates/engine/examples/glm_smoke.rs`
> and `arena_repro.rs` are what compiled in the lock (`cargo build --workspace --examples -v`
> afterwards compiled exactly `rivoli-engine`'s two example crates). The battery's pre-check is
> now `cargo build --workspace --tests --examples`, run twice, refusing unless the second run
> compiles nothing; the workspace's doctests run nothing, so rustdoc compiles nothing either.

**Fixed after the lease, deviceless (re-run owed):** `hc_post` fuses the left product;
`hc_pre`'s two loops use 0b's `rowview::for_strided::<8>` (the C's `#pragma unroll 8`);
`attn_res` indexes `p` only through 16 straight-line constant slots (`each_src`), so its
private segment is 0 B (`llvm-readelf --notes` on the new code object; 144 B before, per the
correctness review's read of the old one); the gate prints a newline before its first verdict line.

### Phase 0a — device arms, lease 2 (2026-11-18, commit `15e7fbf`)

Arms A and R re-run on the lease-1 fixes; **witness CLEAN on both** (0 B witness files, GTT
17 MiB pre-arm, host loadavg ~1.6). Logs `/var/cache/rivoli/scratch/bit-identity/20261118T191409-4152605`
(A) and `…/20261118T191428-4153446` (R).

- **Arm A: 13 of 13 rows BIT-IDENTICAL**, script GREEN (`rows=13 of 13`), NaN-payload-only 0
  on every row. `hc_post` is fixed by the left-product FMA.
- **Arm R: GREEN (expect-red)** — exactly `rope_interleave` and `rope_split_half` red (275/320
  cells each, 53256 / 53002 words), every other row bit-identical. The bit-identity gate's
  red proof is now PAID on the device.
- After-numbers: **`hc_pre` 14.58 → 14.61 µs, ratio 1.002** (from 3.856 — the unroll was the
  whole gap). **`attn_res` 53.14 → 56.51 µs, ratio 1.063** (from 1.383) — above parity by ~6%,
  reported and NOT iterated inside the lease. The rest within 0.86–1.03 (`act_quant_f8` 1.030,
  `rope_interleave` 1.025, `rope_split_half` 1.014).
- **OWED to Phase 1 timing work: `attn_res`'s 1.063** — not closed in this stretch. A static
  read of the ISA gives a lead, not a cause. The two listings were produced with
  `hipcc --offload-arch=gfx1151 -O3 --offload-device-only -S residual.hip` (the tag's file)
  and `llvm-objdump -d --mcpu=gfx1151` of the shipped code object, and each was counted with
  `grep -cE '^\s+(s_|v_|ds_|global_|buffer_|scratch_|flat_)'` over the `attn_res` body.
  - **Body size:** 1617 instructions in Rust's body against 725 in hipcc's.
  - **Static counts, by `grep -c` per mnemonic:** `s_barrier` 10 against 3, global loads 18
    against 3, LDS ops 67 against 16, `v_fma` 127 against 46.
  - **Reading:** hipcc keeps a loop over the sources, and the 16 straight-line `each_src`
    slots expand it.
  - **What is owed:** whether that growth (instruction cache, issue) is the 6% is
    unmeasured, and isolating it is the owed work.

### Phase 0a — the prelude probe rows (2026-11-18, deviceless)

0b's scratch probe families are now rows of the gate. **A module-load reference beside the
shared-object one:** a prelude helper has no launcher, so each is exposed by a probe kernel —
`crates/engine/tests/bit_identity/probes/probe.hip` (0b's file, moved in) built by hipcc
against the TAG's headers, and its twin in `crates/kernels/src/probes` behind a `probes`
feature, built into a SECOND code object by `tests/bit-identity.sh --build-probes` (never
embedded, and never part of a cargo build: the device flags come from the `device-rustflags`
file `build/device.rs` writes, the nested build runs under `env -i` in the crate's directory,
and its clean object is byte-identical — sha256 `3a23b8d7…` — to the one an earlier draft
built inside `build.rs`) — and both are launched through the backend's new `CodeObject`
(path-loaded module, packed kernarg buffer) from one argument list. The Rust probes take one `repr(C)`
block per C parameter list (bytes = the positional kernarg layout); 0b's ISA census re-run
over the in-tree pair: the same single DIFF as their record (`k_i4_wave_r2`, waves/SIMD
16→12; VGPR 104 vs their 103).

- **Rows:** codec (exhaustive 2^32 sweeps), reduce, fp8, i4, f4, vq — 222 groups, each a
  verdict row with the launcher rows' tokens; NaN as a class and 0b's `Out` typing (only f32
  outputs NaN-classified); different fills per side; the compare runs ON THE DEVICE in
  hipcc-built `cmp_*` kernels added to `probe.hip` (the comparer is not the compared).
- **Plants:** 0b's nine. Each is a SMALL PATCH to the real prelude
  (`tests/bit-identity/plants/<family>.patch`), 2–10 changed lines, every hunk marked
  `PLANT`. There is no red-proof branch in shipped source, no copied helper, and no jscpd or
  CodeScene exemption. That was the coordinator's ruling on 2026-11-18. It replaced the draft
  in `604a5c0`, which copied the helpers into a cfg'd `plants.rs` and needed both exemptions.
  - **What `--build-probes [--plant=<family>]` does:**
    - copies `crates/kernels` into `$CARGO_TARGET_DIR/probes/<name>.src` with fresh mtimes;
    - applies the patch with exact context (`patch -F0`);
    - refuses unless the copy's `src/` now differs (diff rc exactly 1) and the checkout's
      `crates/kernels`+`crates/abi` hash is unchanged;
    - deletes the old object before building, and stamps the new one with that hash plus
      the patch's;
    - the arm refuses an object whose stamp no longer matches the tree.
  - **Clean objects:** built from such a copy too, so a plant differs from clean by the patch
    alone.
  - **Red proofs** (temporary fixtures, removed after; `gate-red-proofs.md` §18): a moved
    context line, a no-op hunk, an unknown plant, a tampered stamp, `--build-probes` with
    `--families` — each refused, exit 2.
  - **Built 2026-11-18:** all nine, each object distinct from the clean one. The clean object
    from a copy is byte-identical (`3a23b8d7…`) to the in-place one. The fp8 and nib objects
    are byte-identical to the draft's copied plants (`cae36d52…`, `3bd986b7…`).
  - **Reach:** a patched helper reaches EVERY probe calling it.
    - The formats plant (`f2e4m3_rne`) also reaches `ex_aq8`, so `plant_spec`'s red RE
      includes it and its hold is `ex_aq4`.
    - The actquant patch plants the shared `clamp_nan_propagating`, so both round trips
      redden, as 0b's did.
  - **Small diffs:** every plant could be written as one. Two needed a second hunk only to
    stay warning-free under `warnings = deny` (`reduce`: `shfl_down` unused; `vq`: `quad`
    unused).
- **Judging a plant:** `--plant=<family>` reads the family's row in the script's `plant_spec`
  table — families, the rows that must redden, the rows that must hold (mandatory) — and
  applies 0b's `check-arm.py` rule on the same classifier. vq reaches every vq group (0b:
  15/15), so its hold anchor is the reduce family. `barrier` has a patch but no row: it is a race, and
  its judge, 0b's scratch ISA census (`s_barrier` per kernel), is not in the tree yet. That
  judge is OWED. Five stub arms red-proof the rule (`gate-red-proofs.md` §18).
- **Census:** `every_prelude_probe_has_its_hipcc_twin` — 35 Rust probes, each with a
  same-named `__global__` in `probe.hip` and vice versa (documented one-sided: `cmp_*`,
  `k_vq_wave_r1`, each checked to exist), and every probe name is a string literal some
  harness row launches; a renamed probe reddens it, and so does an unlaunched one (planted
  `"ex_nquad"` → `"ex_nquadX"` in `codec.rs`: exit 101, `probes no harness row launches:
  ["ex_nquad"]`; restored, `cmp` clean, green). The Rust-kernel handle census excludes the
  probes by PATH (`crates/kernels/src/probes/`), so a shipped kernel cannot drop out by
  sharing a probe's name.
- **Divergences from 0b's harness, deliberate:** `k_lut` is judged as `u32` words, strictly
  (0b's was strict too; an f32 reading would have let a NaN-code difference in the table
  pass); fp8's hipcc twin launches with the same 1 KiB of dynamic LDS the Rust twin needs
  (hipcc's LUT is a static array; the bytes are unread); inputs carry 0b's 64 bytes of zero
  tail slack.
- Inputs are this gate's own draws (`Lcg`, 0b's seed classes NORMAL/WIDE/NASTY and shapes), so
  the cells are new; group names use Rust's float formatting (`s=NaN`, not 0b's `s=nan`),
  which `plant_spec`'s regexes are written against.

### Phase 0a — probe lease (2026-11-18, commit `1759536`)

Probe A and the eight judged plant arms ran in that order, once each, under the default
`/var/run/sys-gpu.lock` (`GPU_LOCK` was unset). **The witness was clean on all nine arms:**
every witness file was 0 B, GTT was 17 MiB before each arm, and no arm was discarded.
Probe A took 57 s; the whole run took 281 s. Host loadavg ranged from 4.38 to 14.19 on 32
cores across the run.

Each arm rebuilds the hipcc reference from the same `probe.hip` (`a90bdaef…`). The object
hashes differ only by hipcc's `__hip_cuid_*` symbol: the correctness review extracted
`.text` from three of them and found it identical (`701c2290…`). Probe A's hipcc object is
`7ef34996…`, and the Rust clean object is `3a23b8d7…`. The verdicts below are the
script's GREEN lines, read from stdout; the scripts' own exit codes were not captured
separately.

**Probe A: 222 of 222 rows BIT-IDENTICAL, scripted GREEN.** Logs:
`/var/cache/rivoli/scratch/bit-identity/20261118T204419-491791`.
- **What was compared:** 13 956 cells over the 222 rows. A codec cell is one 2^26-input
  chunk, so the 91 codec rows' 5 572 cells cover 87 exhaustive 2^32 sweeps plus four smaller
  full domains (`ex_bf16f`, `ex_e4m3f`, `ex_e8m0f`, `k_lut`, one cell each). A dot cell is
  one seed's launch.
- **NaN-payload-only: 231 cells raw**, with no value-red cell anywhere. They fall in 62
  groups, all dots:

  | family | groups | cells |
  |---|---|---|
  | fp8 split-K | 17 | 83 |
  | fp8 wave | 15 | 62 |
  | fp4 | 14 | 52 |
  | vq | 12 | 30 |
  | int4 | 4 | 4 |

- **Against 0b:** 0b's scratch A0 had 442 NaN-only cells in 85 groups. Why the count moved
  is inferred, not isolated: this gate draws its own inputs (`Lcg`), and the Rust object is a
  different build. 0b's record already shows NaN payloads moving between two Rust builds of
  the same loop.

**Plant arms: all eight scripted `GREEN (plant <family>)`.** The test binary exited 101 in
each.

| arm | rows | red | must-hold (red) | NaN-only cells | logs `…/bit-identity/` |
|---|---|---|---|---|---|
| formats | 91 | 6: `ex_f2e4m3` (4 cells) + `ex_aq8` at s = 1, 2^-10, 2^20, 0.3, −2 | 12 (0) | 0 | `20261118T204516-496644` |
| actquant | 91 | 18: `ex_aq8`/`ex_aq4` at every non-NaN scale (inf included) | 3 (0) | 0 | `20261118T204606-502170` |
| common | 91 | 4: `ex_swiglu` at all four limits | 45 (0) | 0 | `20261118T204657-512091` |
| situ | 91 | 30: `ex_situ_*` | 5 (0) | 0 | `20261118T204750-520269` |
| reduce | 8 | 1: `k_wave_sum` (64/64) | 7 (0) | 0 | `20261118T204842-527753` |
| fp8 | 60 | 34 | 18 (0) | 84 | `20261118T204845-528098` |
| nib | 48 | 28, every one at a dim with a tail | 20 (0) | 55 | `20261118T204852-528919` |
| vq | 23 | 15, every vq group | 2 (0) | 19 | `20261118T204856-529420` |

- **Red RE check:** for fp8 and vq the red RE matches every row of the family, so
  "no red outside the RE" cannot fail there. Locality rests on the must-hold anchors instead:
  fp8's 18 `block=[12]` groups, and vq's two reduce rows.
- **Against 0b's scratch arms** (compared name by name, by the correctness review):
  - reduce, fp8 and nib: the value-red sets are identical.
  - vq: the same 15 groups.
  - actquant, common and situ: equal counts per name pattern.
  - formats: the five extra `ex_aq8` rows are new. The patch form predicted them: the
    patched `f2e4m3_rne` sits inside `ex_aq8`.
- The barrier plant has no arm; its judge is owed (above).

### Phase 0 merge review — the fix list (2026-11-18, deviceless)

The coordinator's merge review of `864f9b3` (two independent reviewers plus a deviceless run)
found no launcher-semantics defect: guards, order, geometry, LDS, streams and return codes all
match the C. It did find gaps in what holds that true, and in the shape Phase 1's four lanes
will copy. Every MUST item was done; the SHOULD items were done or declined, as below.

**Gates added (each red-proofed deviceless, `gate-red-proofs.md` §18):**
- **Gate rows census** (`kernel_coverage.rs`, `every_rust_kernel_has_a_gate_row_and_a_launchers_row`).
  Three populations, each checked against the others in both directions: the shipped Rust
  kernels, the gate's `ROWS` entries, and the `launchers!` `=>` rows (by the `rivoli_<kernel>`
  host launcher each names). Each `ROWS` entry's row function must also bind
  `c"rivoli_<name>"`, so an entry pointing at another kernel's row cannot stand in. Every
  finding is gathered before one assert, so a defect reaching both populations shows both.
  `touch_region` is the one argued exemption (`UNROWED`, checked to exist). Before this, a
  lane could port a kernel with no row and every census plus the gate would stay green.
  Result: 14 kernels, 13 rowed, 1 exempt.
- **Launcher floor, derived.** The scanned launcher population must equal
  `kernel-inventory.md`'s table, in both directions. The hand-bumped `>= 66` is gone. The
  shipped-kernel floor (`>= 14`) is gone too, replaced by the census above.
- **Binary freshness** (`tests/bit-identity.sh`). The binary is the one cargo names for the
  header's build line, run in the checkout's own root, and it must be Fresh. A stale binary is rebuilt by that call, outside
  the lock, and the arm is refused. A positional (planted) binary is accepted only with
  `--expect-red`. It used to be "the newest by mtime".
- **Aliased rows.** These match how production launches each kernel in place:
  - `swiglu` with `h == g` (`glm/mlp.rs`);
  - `swiglu_clamped_bf16` with `h == g` (`v4/moe.rs`);
  - `situ_glu_f32` with `h == g` (`k3/forward.rs`);
  - `rmsnorm_single` with `y == x` (`glm/attn.rs`);
  - `rmsnorm_centered_single` with `y == x` (`glimmer/forward.rs`).

  The review listed three; its correctness reviewer found the other two production in-place
  launches.

  Each is an untimed sweep over the row's shapes, with the output buffer starting as the
  operand on both sides. These are new cells, and the next device run is their first.

**Phase 1 shape:**
- **Host launchers:** `rk_launchers.rs` became `crates/backend/src/launch/`, one module per
  kernel family (activation, residual, touch). `Geometry` is a named struct (`blocks`,
  `threads`, `lds: LdsBytes`). `launch_raw` takes `(f, geom, stream, KernArgs)`, where
  `KernArgs` is `Params | Packed`: HIP's exactly-one-of rule, made a type.
- **ABI crate:** `rivoli_abi::kernarg` and `geom` became per-family modules, re-exported flat.
  Each struct's size assert sits on the line after it, and each constant's shape assert sits
  beside it. `ROWS_PER_BLOCK` and `FP4_QUANT_BLOCK` moved into `geom`, and the prelude now
  derives its own from them.
- **Invocation files:** `hip_linalg.rs` (799 lines) split by family into `hip_linalg.rs`
  (matvec, embedding, elementwise: 392 lines) and `hip_activation.rs` (gates, quantizers,
  norms, RoPE: 441 lines). A `launchers!` invocation file needs its own jscpd marker, so the
  exemption count is **10** (CLAUDE.md). Both markers were re-measured with both deleted: 11
  clones, 5 inside each file and 1 across them, every one a parameter-list prefix run. So
  each marker is earned, not inherited.
- **Build script:** `build/device.rs` now reruns on the root `Cargo.toml`, because
  `rivoli-abi` inherits its edition and lints from it.

**SHOULD items:**
- **Done:**
  - `gpu::for_i32` replaces 8 hand-written `int` strided loops in the kernels. The ISA moved,
    and the next device run re-scores the bits.
  - The touch launcher reuses `hiprt::based`.
  - `hiprt`'s header documents its four return conventions plus the guard codes.
  - `hip_activation.rs` imports explicitly (no glob).
  - The `§1780` citation became §16 (the /var/run bind defect).
- **Declined:**
  - **Kernarg constructors:** a struct literal names every field at the row. A constructor
    over `g, u, h` (three `*f32`) brings back the positional transposition the struct
    removed.
  - **A `twin!` helper and named shape structs in the gate template:** they would touch
    every row adapter before a lease, for readability rather than gate strength. Left for the
    first Phase 1 lane that adds rows, if it finds the template in the way.
  - **`CodeObject` behind a test-only feature:** a backend feature joins the checked feature
    matrix (`matrix.rs`), and `CodeObject` has no engine caller to leak into.

**Text corrections:**
- `residual.rs`: now cites 1.002 and 1.063, and names `each_src` as the suspect for the
  remaining 6%.
- The prelude's "probes in scratch" note.
- `kernels/lib.rs`: "every kernel through that path" now excludes `touch_region`.
- `module.rs`: the compile-error claim is narrowed to the static's type.
- `stage.rs`: "negative" corrected to "positive".
- CLAUDE.md's bit-identity bullet.

### Phase 0 merge review — the device lease (2026-11-18, commit `3706fdc`)

Six arms ran in order, once each: A, R, P (probes plus 8 plants), W, D, T. The lock was the
default `/var/run/sys-gpu.lock`. **No arm was discarded.** Every witness file was 0 B, and GTT
read 18702336 B (17.8 MiB) before each arm.

For D and T, `mem_info_gtt_used` was also read immediately AFTER the arm, because llama-swap's
embedding model reloads on demand and the witness samples GTT only before an arm. The rise was
0 on both.

The wrapper that does that read (`gpu/gttwrap.sh`) was self-tested once, wrapping `true`, at
22:09, inside W's window (`lease4/selftest.gtt`). That was a sysfs read, not a device command.

Evidence is in `/var/cache/rivoli/scratch/rk-infra/lease4/` and in the log directory each arm
names. Exit codes for A, R and P were not captured separately: `bit-identity.sh` exits 0 only
after printing a GREEN line, so those codes are implied by the scripted verdict. The codes for
W, D and T were captured.

| arm | scripted verdict | rc | witness | notes |
|---|---|---|---|---|
| A (launcher rows) | GREEN: 13/13 BIT-IDENTICAL | 0 (implied) | 0 B | logs `…/bit-identity/20261118T220249-907555` |
| R (rope plant, rebuilt at `3706fdc`) | GREEN (expect-red): exactly `rope_interleave`, `rope_split_half` | 0 (implied; binary 101) | 0 B | 275/320 cells red on each, as before |
| P: `--probes` | GREEN: 222/222 | 0 (implied) | 0 B | 231 NaN-payload-only cells, the same as the probe lease |
| P: 8 plants | `GREEN (plant <family>)` ×8 | 0 (implied) each | 0 B each | red counts 6/18/4/30/1/34/28/15, the same as the probe lease |
| W (full battery, `--test-threads=1`) | 707 passed, 1 failed, 2 ignored | 101 | 0 B | the failure is CodeScene, 31 files as on the base; 0 units compiled inside the lock; about 17 min |
| D (`determinism-glm.sh`, `--arena-refresh`, ngen 1024) | GREEN: ids identical to 1024 | 0 | 0 B / 0 B | GTT rise 0; decode 376 / 346 s (2.72 / 2.96 tok/s); about 20 min |
| T (trace build, 32-token GLM decode) | GREEN | 0 | 0 B | the first 4 of 32 ids `[13041 1052 0 358]` match the reference; 0 `READ-BEFORE-WRITE`; GTT rise 0; 4612 decode misses |

**What this settles:**
- **A: the outputs after the `for_i32` loop rewrite, and the five aliased rows' first device
  cells, are bit-identical.**
  - The aliased rows are `swiglu`, `swiglu_clamped_bf16` and `situ_glu_f32` (1024 cells each,
    was 512) and `rmsnorm_single` and `rmsnorm_centered_single` (768 each, was 384).
  - Their in-place moved counts are 512 for each gate row, 320 for `rmsnorm_single` and 384 for
    `rmsnorm_centered_single`; a row with any in-place cells must have a non-zero count to be
    green.
  - The aliased rows still have no red proof: nothing planted here breaks the in-place path.
- **The gate kernels' timing swings between arms, cause unknown.** In A the three gate rows timed
  1.12–1.17× hipcc (swiglu: hipcc 2.16 µs, Rust 2.47 µs). In R, which runs the same tree minus
  rope, the same rows timed 0.96–0.99×. Lease 2's R read up to 1.21× on the same rows. The swing
  is mostly on the Rust side, and there is no A-vs-A control. Nothing here is a measured
  regression, and nothing here rules one out.
- **`hc_pre` and `attn_res` after `for_i32`:**
  - `hc_pre`: 1.006×.
  - `attn_res`: 1.042× in A and 1.045× in R, against 1.063× at lease 2. It is still above
    parity and still owed to Phase 1.
- **D: the rustc-built `touch_region` still mitigates GLM's nondeterminism at 1024 tokens.**
  - If the touch were broken, P(detect) = 1 − exp(−2n/rate) (`glm-nondeterminism.md`'s
    table) at n = 1024 is:
    - 97% at the conservative rate (1 per 578);
    - 99.89% at the matched rate (1 per 299);
    - about 52% at the conservative interval's pessimistic end (1 per 2804).
  - It bounds the rate; it does not prove determinism.
  - The engine logged `slot_stalls 0`, the INV-9 soundness precondition.
  - The script's own scratch is under `${TMPDIR:-/tmp}` (not this file's to change); the
    evidence was copied to `lease4/D-scratch/`.
- **T: `fill_u32` (`hipMemsetD32Async`) into the routed pool's `hipMemMap`'d slots is
  accepted.**
  - Under `--features trace`, `routed.rs` poisons each admitted slot through it. The result is
    propagated with `?`, and so is the `device_sync` that follows, so a refused fill fails the
    run.
  - The limit: there is no per-fill counter, so "on every admission" rests on the code path,
    not on a log line.
  - Only the first 4 ids have a recorded reference (`smoke-glm.sh`), so a poison leaking into
    compute after position 4 would not be seen.

### Phase 0b — the prelude (2026-11-18)

> **MERGED 2026-11-18 into `wave/rk-infra`:** 0a's manifest, `lib.rs` and build integration
> kept (0b's clippy deny entries adopted); `sync_threads` is now 0a's convergent-intrinsic
> form (record under Phase 0a, census re-run unchanged); 0a's inline helpers deleted in
> favour of this prelude.

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

**Device results, 2026-11-18, rh-anine.** One lease, running `run-arms.sh` exactly as
posted. `sha256sum -c artifacts.sha256` passed immediately before launch. Every arm's
witness is CLEAN: the `/var/cache/rivoli/scratch/rk-prelude/probe/witness/witness-*`
files are all empty, GTT was 17 MiB before every arm, and loadavg 2.2–2.8 on 32 cores.

Evidence, all under `/var/cache/rivoli/scratch/rk-prelude/` and every file below hashed in
`probe/arms/outputs.sha256`:
- raw outputs: `probe/arms/<arm>.out` and `probe/arms/run.log`;
- verdicts under the owner's rule (below): `probe/arms/verdicts-nan-class.txt`, written by
  `check-arm.py --byte-exact-rc` over the same raw outputs (no device re-run);
- the checker's red proofs: `redproof-checker/verdicts.txt`.

`run-arms.sh` reported **UNEXPECTED (rc 1)** under the byte-exact rule it ran with, for
one reason: NaN payloads. A red cell is `nan_payload_only` when every differing f32 word in
it is NaN on BOTH sides. NaN against non-NaN counts as value-red, so this class cannot hide
a value difference. Which outputs are classified is stated under the owner decision below.

| arm | cells | red groups (byte-exact) | value-red cells (where) | NaN-only cells | value verdict |
|---|---|---|---|---|---|
| A0 hipcc vs rustc | 13 956 | 85 / 222 | **0** | 442 | identical |
| R-formats | 5 572 | 1 / 91 | 4 (`ex_f2e4m3` only) | 0 | PASS |
| R-actquant | 5 572 | 18 / 91 | 38 (`ex_aq8`/`ex_aq4`; s = NaN held) | 0 | PASS |
| R-common | 5 572 | 4 / 91 | 54 (`ex_swiglu` only) | 0 | PASS |
| R-situ | 5 572 | 30 / 91 | 60 (`ex_situ_*` only) | 0 | PASS |
| R-reduce | 512 | 1 / 8 | 64 (`k_wave_sum`; `wave_max`, `block_sum` held) | 0 | PASS |
| R-fp8 | 3 840 | 40 / 60 | 1728 (34 of the 42 block ≥ 4 groups) | 103 | PASS on values |
| R-nib | 3 072 | 36 / 48 | 1508 (28 groups, all at dims with a tail) | 48 | PASS on values |
| R-vq | 960 | 15 / 15 | 897 (every group) | 21 | PASS |
| R-barrier | 512 | 0 / 8 | 0 | 0 | race not observed (census is its gate) |

- **Every non-NaN output is bit-identical.** That covers all 87 exhaustive 2^32 codec
  sweeps (including every NaN input, whose outputs match bit for bit), every reduction, and
  all 64 seeds of every dot shape.
- **The A0 reds are only in the dots:** fp8 35 groups, int4 23, fp4 15, vq 12. Every
  group's FIRST red seed is in the nasty class (7/15/23/31/39/55; `harness.cpp`: seed%8 ==
  7, which is 8 of the 64 seeds), and no group has more than 8 red cells. That fits "nasty
  seeds only", but the harness records only the first seed per group, so this is not
  proven.
- **Cause (read from the ISA before the run, INFERRED rather than isolated).** The
  compilers order commutative operands differently: hipcc emits `a1 + a0` before the nibble
  `wave_sum` and `l0·x0` in the fp8 quad (`probe/hip.dis` vs `probe/rust.dis`, kernels
  `k_f4_wave_r1`, `k_i4_wave`, `k_fp8_wave_r1`). That is consistent with RDNA keeping one
  fixed operand's NaN payload. **vq's 12 groups (21 cells) have no site identified.**
- **The order is not even stable across Rust builds.** The nib plant edits only the tail,
  yet it changed NaN payloads outside it:
  - int4 no-tail groups: 76 NaN-only cells in A0, 0 in R-nib;
  - fp4 no-tail groups: 63 → 25.

  R-fp8's must-hold groups (block 1/2) carry exactly A0's NaN cells.
- **Every plant is value-red only where its CPU proof says.**
  - Under the byte-exact rule, the checker FAILed R-fp8 and R-nib, on NaN-only cells in
    must-hold groups (value-red there: 0).
  - The 8 fp8 block ≥ 4 groups that held are the tiny shapes (i_dim 4 and 36). There each
    lane folds at most one quad into a zero accumulator, and `fma(s, q, 0) == 0 + s·q`
    exactly, so the defect cannot show.
  - Judged under the owner's rule below, every judged arm PASSes and A0 has 0 red groups
    (`probe/arms/verdicts-nan-class.txt`). R-barrier is informational and not judged.

**Owner decision 2026-11-18: NaN as a class.** A word that is NaN on BOTH sides counts as
equal; every other value stays byte-exact. The decision rests on three things:
- In A0, all **442 of 442** mismatched cells were NaN-payload-only, with 0 value-red.
- No source spelling controls each compiler's commutative-operand order, and the nib arm
  shows it moving between two Rust builds of the same loop.
- No consumer reads a NaN payload; every one treats NaN as a class. The argmax skips NaN
  (strict `>`, `crates/engine/src/glm/decode.rs:17-19`), and decode bails if the winning
  logit is non-finite (`glm/decode.rs:71`; likewise `v4/decode.rs:62`, `k3/decode.rs:53`,
  `glimmer/forward.rs:595`).

How it is applied:
- **Both verdict tools use the rule.** `harness.cpp` (`struct Group`) and `check-arm.py`
  judge a group red iff `mismatched_cells > nan_payload_only`.
- **The raw counts stay visible.** Every group line still prints the byte comparison's
  `mismatched_cells` and `nan_payload_only`, and `check-arm.py` prints the arm's total
  NaN-payload-only cells, so a payload change is still observable.
- **Strict outside f32.** Every sweep call site states its output type (`enum class Out`),
  and only `Out::F32` is NaN-classified, so a forgotten case stays strict. That covers the
  1-, 2- and 8-byte outputs and the u32-code sweeps `ex_f2e2m1`/`ex_blk_shift`/`ex_nquad`.
  The f32 `k_lut` cell is also judged strictly. Every `pair()` kernel (dots, reductions)
  writes f32. The pre-rule binary that ran classified by size alone; that made no
  difference here, because the integer sweeps had 0 mismatched cells in every arm.
- **`check-arm.py` red-proofed deviceless** (`redproof-checker/verdicts.txt`):
  - one A0 cell edited to carry a single value-red word → FAIL at harness rc 0 and rc 1;
  - every NaN-only count zeroed (the byte-exact view) → FAIL, 85 red groups, at rc 0 and
    rc 1;
  - the real A0 → PASS only with `--byte-exact-rc` (the pre-rule rc 1 is accepted when the
    RESULT line's red count equals the groups with any mismatched byte) and FAIL without;
  - an expect-red arm with no RED_RE → FAIL.

  **The harness's own copy of the rule is UNVERIFIED-NEEDS-GPU.** It has not executed. Its
  first lease should reproduce A0 = rc 0, with R-actquant as its NaN-against-number red
  proof (18 red groups, `nan_payload_only=0`).
- **Known cost:** a NaN that propagates through a dot lane masks any upstream value
  difference in that lane, as it already did under byte compare.
- **Provenance.** The harness was rebuilt with the rule after the lease. `artifacts.sha256`
  names the binary that ran (`68a0…`), but its exact source was overwritten and is NOT
  preserved. `probe/arms/outputs.sha256` names the rebuilt binary and its source.

**Measured sweep rate (first measurement).** These are arm wall-clock spans from
`arms/*.out` mtimes (1 s resolution) against `arms/start.epoch`, so they include setup and
are LOWER bounds on throughput:
- **Codec arms:** 48–49 s each for 87 full 2^32 sweeps plus 4 small ones, i.e. ≈ 0.55 s per
  2^32 sweep. Each input is evaluated by both kernels; both buffers are refilled and compared
  on the device, 6 × element-size bytes of traffic. Over the 1.49·10^12 compared bytes of a
  codec arm that is ≈ 185 GB/s.
- **A0:** 50 s.
- **Whole run:** 247 s to the last arm's output, 267 s to `arms/end.epoch`. The posted
  READY-FOR-GPU estimate was 25–35 min.

### Phase 1 L2 (moe, moe_f4, recurrent)

**Deviceless half, 2026-11-19 (branch `wave/rk-l2-moe`).** `recurrent.hip`, `moe.hip` and
`moe_f4.hip` are Rust: 18 kernels in `crates/kernels/src/{recurrent,moe,moe_f4}.rs`, their 10
launchers in `crates/backend/src/launch/{recurrent,moe}.rs`, `=>` rows in `hip_attn.rs` and
`hip_blocks.rs`. The three `.hip` files are deleted and out of `build.rs`.

**The shape the port took:**
- **One kernel argument block per range FORMAT, not per kernel** (`rivoli_abi::kernarg::moe`).
  A range launcher issues gate/up then down over the same span, descriptors, `h` and routing
  weights, so both kernels take one struct (`MoeRange<D>`, plus the VQ codebooks or the fp4
  parameters).
- **`ExpertDesc`/`ExpertDescF4` live in `rivoli_abi`.** The kernels read the definition the
  engine writes. `rivoli_backend::hip` re-exports both, so no caller moved.
- **`launch_gdn_recurrent_f32` is a DSL row again.** A `=>` row's call expression is free-form,
  so its `GdnBufs` group reaches the kernel whole, as `DeltaInputs`. The hand-written wrapper
  and its `extern` are gone.
- **The C export `rivoli_act_quant_f8` is gone.** `moe_f4.hip` was its only C caller. The V4
  launcher now calls the Rust function between its passes.
- **No kernel reads `blockDim`.**
  - The delta rule takes its block as `head_dim`.
  - The two masked kernels re-derive theirs with `rivoli_abi::geom::reduce_block`, which their
    launchers also launch with. It is `next_pow2`, as `u32::next_power_of_two`.
  - The passes stride by `ROWS_PER_BLOCK` waves.
  - The drains and the conv stride by `ELEMENTWISE_BLOCK`.

**Fused sites, read from hipcc's IR and confirmed by the ISA census:**
- The delta rules' three per-loop pairs (`u += k·sv`, `s + k·dv`, `o += q·sv`).
- The conv's `acc += w·win`.
- The drain's `x += gain·f`.
- Everything inside the prelude's dots.

Nothing else pairs a multiply with an add.

**The atomic.** hipcc emits `atomicrmw add … syncscope("agent") monotonic`. `core`'s
`AtomicU64::fetch_add(Relaxed)` is SYSTEM scope. Both lower to the identical encoding,
`global_atomic_add_u64 … off` = `DD0E0000 007C0200`, with no cache bits. The output is integer,
so it does not depend on the order the waves land in.

**L2 obligation 1 — fp4 global vs flat loads: the census half is CLOSED deviceless, without a
fallback.** Phase 0b's L2 obligation (§ Phase 0b, *Not expressible: `gu8p`*) was "a
flat-vs-global load census of `moe_gateup_f4`/`moe_down_f4` plus a timing row against hipcc. A
buffer_load fallback ships only with a measured timing at parity." No fallback is needed; the
timing row is owed to the lease (below). The census of the real kernels, load mnemonics against
hipcc+`gu8p` (the down kernels counted; the gate/up kernels had no flat load in any build):

| kernel | hipcc | Rust, descriptor by reference | Rust, descriptor by value (shipped) |
|---|---|---|---|
| `moe_down_f4` | 6 `global_load_b32` + 7 `global_load_u8` | 5 `flat_load_b32` + 7 `flat_load_u8` + 1 `global_load_b32` | identical to hipcc |
| `moe_down_f4_weighted` | 7 `global_load_b32` + 7 `global_load_u8` | 5 `flat_load_b32` + 7 `flat_load_u8` + 2 `global_load_b32` | identical to hipcc |
| `moe_gateup_f4` | no flat load | no flat load | identical to hipcc on every counted mnemonic |
| `moe_gateup_f4_situ` | no flat load | no flat load | loads identical; one `v_fmamk` spelled `v_fmac` |

- **The cause.** 0b's finding reproduced exactly on the real `moe_down_f4`: 5 `flat_load_b32`
  plus 7 `flat_load_u8`, against hipcc's `global_load`. The cause is not the atomic. With the
  atomic replaced by a plain store, or removed entirely, the loads stayed flat. The cause is how
  the descriptor was handed to the dot. As `&*descs.add(e)`, the pointer loads reached
  `AMDGPUPromoteKernelArguments` without `!amdgpu.noclobber` (read from the LTO
  `precodegen.bc`, `--save-temps`). Copied by value, as the C's
  `const ExpertDescF4 d = descs[e]`, they were marked noclobber and promoted.
- **The ISA census after the change** (0b's `census.py`, extended to count stores, atomics and
  scratch, at `/var/cache/rivoli/scratch/rk-l2-moe/census.py`: the Rust code object against
  `hipcc --offload-arch=gfx1151 -O3 -fPIC --offload-device-only --no-gpu-bundle-output` of each
  tagged file, both disassembled with `llvm-objdump -d --mcpu=gfx1151`; output
  `census-final.txt` beside it):
  - **fp4 is load-type identical to hipcc+`gu8p`.** `moe_down_f4`, `moe_down_f4_weighted` and
    `moe_gateup_f4` count SAME on every FP, load and atomic mnemonic.
  - **VQ and int4 go one better.** The six GLM down/gate-up kernels now use `global_load` where
    hipcc, with no `gu8p` on those formats, emits `flat_load`. For example, `moe_gateup_i4` has
    22 `flat_load_b32` in hipcc's code and 0 in Rust's.
- **The timing row against hipcc at real shapes is the device lease's.**

**L2 obligation 2 — `moe_*_i4` occupancy:**

| kernel | hipcc VGPR | Rust VGPR | waves/SIMD |
|---|---|---|---|
| `moe_gateup_i4` | 93 | 93 | 16 / 16 |
| `moe_gateup_i4_r2` | 127 | 129 | 10 / 10 |
| `moe_down_i4` | 89 | 88 | 16 / 16 |
| **`moe_down_i4_r2`** | **95** | **100** | **16 / 12** |

- Only the R=2 down pass crosses a granule. It is the same edge 0b's probe `k_i4_wave_r2` showed:
  96 → 103 there, and hipcc's R=2 int4 dot sits right at 96.
- It is a prelude property (`dot_nib_wave_r::<2, 4, I4>`) and not one this lane can move without
  re-scoring every prelude caller.
- Its cost is the TIME line at GLM's real (6144, 2048) nrow=2 shape. That line is the lease's.

**Other ISA differences, none a value difference:**
- `rmsnorm_gate_heads_f32` counts SAME.
- **The delta rules** carry more loop copies, because `for_strided::<4>` unrolls its remainder.
  They also carry one extra `v_ldexp`, because `d as f32` converts from a u64 where the C
  converts from an i32. That conversion is exact for d ≤ 1024.
- **`gated_delta_recurrent_f32`** evaluates `exp(a_log)` on the VALU where hipcc used the SALU.
- **`short_conv_silu_f32`**: rustc vectorises the in-place shift loop to b128, which is legal for
  a forward shift. VGPR 9 → 25, still 16 waves.
- **`moe_acc_drain{,_to}`**: rustc unrolls the rows loop.
- **`moe_gateup_f4_situ`**: one `v_fmamk` is spelled `v_fmac` with the literal in a register.
  That is the same fused op.

**Gate:**
- **Ten launcher rows** in `crates/engine/tests/bit_identity/{recurrent,moe,moe_f4}.rs`, each
  over 64 seeds per shape.
- **Real shapes:**
  - GLM 6144 × 2048 at nrow 1 and 2;
  - V4 4096 × 2048, as one streamed expert and as the six-expert resident batch;
  - K3 3584 × 3072, plus 96 heads × 128;
  - Qwen 16/48 heads × 128;
  - the anchors' 32 and 20.
- **Small, ragged and padded widths** alongside them.
- **Range cells are real descriptor arrays** into weight and scale arenas, with a per-expert
  shift that alternates 4-byte alignment, so odd experts take the dots' scalar tail on every row.
- **Scales carry NaN, ±inf, −0 and subnormal codes**: bf16; f32; and e8m0 `0xff`/`0x00`/`0xfe`.
- **Routing weights carry exact ±0 and a NaN.**
- **`h`, the recurrent state and `acc` start identical on both sides.** `acc` is compared as
  strict integer words, through a new `Out::Words`, because under NaN-as-a-class a negative
  i64's high half is a NaN pattern.
- **Guard rows pin every code**, and a pair per adjacent two in each launcher's order. They run
  against a live guard descriptor array and an offset guard buffer, so a regressed guard reads
  RED instead of faulting the GPU.
- **Every in-place output must move in some cell** (`Tally::moved_by_out`), so a range row
  whose accumulator never changed cannot pass on the strength of its `h` having moved.
- **Routing-weight specials sit inside the range and always leave one live weight**, and the
  gate/up scale and codebook edges ride a seed class (`seed % 8 == 3`) apart from the NaN-`x`
  seeds, which would otherwise turn every R=1 gate/up dot NaN and hide them.
- **Three plants**, each shown adversarial on CPU (`bit_identity/plants.rs`):

  | plant | defect | CPU seeds moved | device arm |
  |---|---|---|---|
  | `recurrent` | the rank-one update unfused | 64/64 | `--expect-red=gated_delta_recurrent_f32,gdn_recurrent_f32` |
  | `moe` | the range kernels' SiLU in multiply form | 64/64 (1101/4096 gate rows) | `--expect-red=moe_expert_range,moe_expert_range_i4` |
  | `moe_f4` | K3's pass-2 bf16 moved after the routing weight | 64/64 (4096/4096 rows) | `--expect-red=moe_expert_range_f4_situ` |

- **The census's `LAUNCHED_BY` rows** (14: the 12 range-kernel phases under their four range
  launchers, and the two single-kernel drains under the tag's `_s` launcher names, whose gate
  rows are renamed `moe_acc_drain_s`/`moe_acc_drain_to_s`). Each launcher's own body launches
  its kernels' statics. Plant: the `(moe_down_i4_r2, moe_expert_range_i4)` entry's launcher
  REPLACED with `moe_expert_range` reddened the census (`rivoli_<launcher> does not launch the
  kernel's static handle`); restored by copy, `cmp` rc 0. A static of the wrong format inside
  the right launcher is not this census's to see: the bit-identity rows own that defect.
- **No file-scope `#pragma clang fp contract(off)`** in the three files or the headers they
  include; the two function-scope ones (`formats.hpp`'s e4m3 subnormal, `actquant.hpp`'s
  `fast_round_scale`) are the prelude's, ported as plain Rust (which never contracts).
- **Device arms and TIME lines are owed to the lease.**

**Device lease, 2026-11-19, commit `60ac104`, two arms, rh-anine.**

> **CORRECTED 2026-11-19 (superseded by the `79e218c` lease below).** In this lease's range
> rows, the up rows of every expert past the first read the down scales' planted edge codes
> (bf16/f32 NaN or ±inf; e8m0 NaN or 2^127). One `h` word was therefore non-finite on both
> sides, and every down contribution took the same clamp bound. So for every shape whose live
> experts sat past expert 0 — every real GLM, V4 and K3 shape — the down-pass comparisons were
> vacuous, and the MoE TIME lines below timed a down pass on NaN `h`. Its 23/23 is evidence
> for every non-range row and, almost everywhere, for the gate/up pass. Cause and fix are under
> "After the lease".
 Both arms ran through
`tests/bit-identity.sh`, detached with `setsid` (own PGID). The waits were on the result files,
not on process absence. Both witnesses CLEAN: the `witness-bit-identity` files are 0 B and GTT
was 17 MiB before each arm (the script reads GTT only pre-arm). Host loadavg, from the script's
own samples: 0.98 → 1.88 on A and 1.33 → 2.14 on R, on 32 cores; L1 was compiling deviceless
during the lease.

| arm | binary | scripted verdict | rc | wall | logs `…/bit-identity/` |
|---|---|---|---|---|---|
| A | `bit_identity-efca1da5680a0280`, the script's freshness check passed at `60ac104` | GREEN: 23/23 BIT-IDENTICAL | 0 | 158 s | `20261119T125122-2978620` |
| R | `…/rk-l2-moe/plant/bit_identity-l2-plants`: `git archive 60ac104` plus the three `plants/*.patch`, built into `target/rk-l2-moe-plant` | GREEN (expect-red): exactly the five planted rows | 0 (binary 101) | 161 s | `20261119T125453-2997362` |

- **A.** `moe_expert_range` reports 1542 NaN-payload-only words: NaN in both twins, payloads
  different, equal under the owner's NaN-as-a-class rule. Every other L2 row has 0.
- **R.** Every unplanted row held: V4's `moe_expert_range_f4`, plus `short_conv_silu_f32`,
  `rmsnorm_gate_heads_f32` and both `_s` drains. Cells red per plant:

  | plant | cells red |
  |---|---|
  | `recurrent` | 287/384 (`gated_delta`), 320/384 (`gdn`) |
  | `moe` | 360/384 on each GLM row |
  | `moe_f4` | **48/320, all in the small (32, 32, 0, 1, 1) cell** |

- **OPEN — the `moe_f4` plant does not reach K3's real shapes.** *(CLOSED 2026-11-19: a
  harness defect, see "After the lease"; the `79e218c` lease's R is red at both K3 shapes.)*
  - **What the log shows.** The row's first six mismatch notes (`Tally` keeps six, in run
    order) are all from the LAST shape, `(32, 32, 0, 1, 1)`, and that shape runs after the two
    real K3 shapes. Every other K3 cell compared identical under the plant: the two real
    shapes, `(96, 64, 0, 3, 1)` and `(160, 96, 1, 2, 1)`. 48 is 64 seeds less the 8 NaN-`x`
    and 8 edge seeds.
  - **Why this contradicts the CPU model.** Every shape carries at least one live,
    non-power-of-two routing weight, and the CPU model moved 4096 of 4096 rows.
  - **Lead (untested).** The one red shape is the only one whose live weight is expert 0. The
    blind shapes' live weights sit at expert index ≥ 2.
  - **Consequence.** The gate is not yet shown able to catch a bf16-after-weight defect in
    K3's pass 2 at production shapes.
  - **Owed.** Per-shape red counts from a re-run of R. The `Tally` gap this exposed is that it
    keeps no per-shape count.

**TIME lines at the hot shapes.** Each is the minimum of three 200-launch batches, Rust ÷ hipcc.
The lane's threshold, from the coordinator's brief, is "≤ ~1.05 parity". There is no A-vs-A
control in this lease. Arm R's unplanted rows are a second sample: the same gate rows swung
0.999 → 1.194 (`swiglu`) and 1.004 → 1.159 (`situ_glu_f32`) between the two arms, the swing the
Phase 0 merge review recorded. So differences of a few percent here are not resolved.

| row (shape) | hipcc µs | Rust µs | A ratio | R ratio (unplanted rows) |
|---|---|---|---|---|
| `gated_delta_recurrent_f32` (96, 128) | 28.98 | 47.73 | **1.647** | — (planted) |
| `short_conv_silu_f32` (12288, 4) | 4.31 | 3.96 | 0.919 | 0.924 |
| `rmsnorm_gate_heads_f32` (96, 128) | 2.89 | 2.70 | 0.932 | 0.938 |
| `gdn_recurrent_f32` (16, 48, 128, 128) | 45.48 | 45.45 | 0.999 | — (planted) |
| `moe_expert_range` GLM (6144, 2048) nrow 1 | 110.27 | 113.03 | 1.025 | — (planted) |
| `moe_expert_range` GLM nrow 2 | 122.55 | 125.55 | 1.025 | — (planted) |
| `moe_expert_range_i4` GLM nrow 1 | 243.28 | 245.67 | 1.010 | — (planted; down unchanged: 1.009) |
| `moe_expert_range_i4` GLM nrow 2 | 307.01 | 319.99 | 1.042 | — (planted; down unchanged: 1.014) |
| `moe_acc_drain_s` (12288, 2) | 3.95 | 3.88 | 0.983 | 0.973 |
| `moe_acc_drain_to_s` (12288, 2) | 3.88 | 3.76 | 0.969 | 0.983 |
| `moe_expert_range_f4` V4 (4096, 2048), 1 expert | 54.09 | 55.20 | 1.021 | 1.019 |
| `moe_expert_range_f4` V4, 6-expert batch | 775.17 | 791.57 | 1.021 | 1.036 |
| `moe_expert_range_f4_situ` K3 (3584, 3072), 1 expert | 60.48 | 62.55 | 1.034 | — (planted) |
| `moe_expert_range_f4_situ` K3, 4-expert batch | 714.82 | 726.09 | 1.016 | — (planted) |

- *(CORRECTED 2026-11-19: reopened for K3 by the `79e218c` lease, below; these samples timed a
  down pass on NaN `h`.)* **Obligation 1, timing half.** The fp4 rows run at 1.019–1.036 across both samples, within
  the lane's threshold, with the down kernels' loads at `global_load` (census above). No
  fallback is needed. Phase 0b's L2 obligation is recorded as met: census identical, timing
  within threshold on two samples, no A-vs-A floor.
- *(CORRECTED 2026-11-19: see the `79e218c` lease, where A read 1.070.)* **Obligation 2.** The GLM int4 range at 6144 × 2048, nrow 2, read 1.042 in A. In R it read
  1.014: R's `moe` plant changes only the gate/up SiLU, so `moe_down_i4_r2` was the same
  kernel in both arms.
  - The 16 → 12 waves crossing (obligation-2 table above) is a candidate cost, not a measured
    one. The launcher row times both passes together.
  - Both samples are within the threshold.
- **Atomics.** No CAS loop: the accumulation is the native `global_atomic_add_u64` (above).
  *(The ≤ 1.036 bound below is CORRECTED 2026-11-19 by the `79e218c` lease's 1.070 rows.)*
  Every MoE row except the int4 nrow 2 one is ≤ 1.036.
- **The one slow row: `gated_delta_recurrent_f32`, 1.647 in A.** The lead below comes from a
  static ISA read and is UNTESTED.
  - **What the ISA shows.** In hipcc's state loop, each 4-way unrolled step issues all four
    `global_load_b32` before its stores and consumes them under `vmcnt(3)`/`(1)`/`(0)`. In
    Rust's, each load is issued only after the previous step's store and is consumed at once
    (`vmcnt(0)`): one request in flight.
  - **The precedent.** That disease is the one `recurrent.hip`'s `#pragma unroll 4` note cites
    from `dot_f4_wave_r` (+33.2%, 146.63 → 195.27 GB/s). That note marks the magnitude
    unmeasured for this kernel.
  - **Lead: hipcc carries a value range on the stride `d`, and Rust does not.**
    - hipcc reads `d` from `blockDim.x`, and its loop masks it with `0x7fc` (bounded below
      2048), which lets it prove the rows `i·d + t` and `(i+1)·d + t` distinct and hoist the
      loads.
    - The Rust kernel reads `d` from the kernel argument (Phase 0c's rule), with no range.
  - **Control.** In `gdn_recurrent_f32` BOTH compilers serialize the same way: load,
    `vmcnt(0)`, store, step after step. Its stride `dv` is a kernel argument in the C too, so
    the twins run 0.999 / 1.001. The two runs agree, but they do not show the fix.
    `gdn`'s dominating `t >= dv` exit (which implies `dv ≥ 1`) did not make hipcc hoist, so
    an `assert_unchecked(d != 0)` is not shown to work.
  - **Secondary, a type choice.** The Rust loop counter is 64-bit (`usize`:
    `s_add_u32`/`s_addc_u32`, `v_cmp_ge_u64`) where the C's is `int`.
  - **Status: reported, not fixed.** Two candidate fixes: loading each unroll group before
    storing it, which is the known-good form; or a range fact on `d` (≤ 1024, which the
    launcher's guard 1002 guarantees), which is unproven. Either one is to be checked
    deviceless against the `vmcnt(3)/(1)/(0)` pattern, then re-timed on the device.
- **The 13 Phase 0 rows** in A lie between 0.853 and 1.014, except `attn_res` at 1.048, which
  is already owed.

**After the lease, deviceless (2026-11-19): the `moe_f4` gap was a HARNESS defect, now fixed.**
- **Cause, found by the CPU model before any re-run.** The range rows' arenas packed `[gate, up,
  down]` back to back, and each expert's spans shift by `e · W_SHIFT`/`e · S_SHIFT`. So the
  last up rows of every expert past the first read the first bytes of the DOWN scales, where
  the edge codes sit (e8m0 `0xff` NaN, `0xfe` = 2^127; the bf16/f32 NaN and ±inf). One
  non-finite `h[j]` makes every down dot NaN, and `moe_fixed` clamps every NaN contribution to
  the same bound on both sides. A model of the real (3584, 3072, 2, 1, 1) cell printed
  `dv=NaN` on all 64 down rows of expert 2, seed 1.
- **Reach.** In every range row (VQ, int4 and both fp4), at every shape whose live experts sat
  past expert 0, pass 2 compared nothing real: GLM's real shapes start at expert 3. Arm A's
  23/23 is therefore NOT evidence for the down kernels at those shapes. Pass 1 (`h`) was
  compared genuinely almost everywhere (the experts past the first carried one NaN `h` word on both sides); that is why the `moe` plant reddened 360/384.
- **Fix** (`bit_identity/ranges.rs`, `Layout::new`). Each projection is padded by every
  expert's shift, so no span crosses into the next projection; the down edges now poison only
  their own down row. The range inputs moved to `ranges.rs` (deviceless) so the CPU models
  build the very cells the rows launch.
- **Live weights on every expert.** The routing-weight specials (±0, 0.3/NaN) now rotate with
  the seed, so each expert of a range — the first and the last included — is live on some
  seeds: on (live − 2)/live of them (1/2 for K3's 4-expert batch, 1/3 for a 3-expert range).
- **Projection bases stay aligned.** The padding is rounded to 16 bytes, so only the deliberate
  per-expert shift misaligns a span.
- **Per-shape tally.** Every row now prints `red by shape: <shape> red/cells, …`
  (`Tally::by_shape`), not only its first six mismatch notes.
- **The plant's CPU model is now the real cell.** `moe_f4_late_bf16_plant_reaches_k3_production_shapes`
  runs the actual K3 cells (`ranges::cell_inputs`, the row's betas) through both passes. The bf16-after-weight plant moves 48/48 of the
  seeds on which pass 2 is live, at BOTH (3584, 3072, 2, 1, 1) and (3584, 3072, 0, 4, 1). The
  excluded 16 are the NaN-`x` and edge seeds. *(CORRECTED 2026-11-19: on an edge seed, an
  expert past the first sits beyond the gate scales' planted NaN, so pass 2 can be live there.
  The device reddened 1–2 such seeds per K3 shape; the model filters those seeds out by class
  and so makes no prediction about them.)*
  Red proof of the model: with the padding removed (copied aside, edited, run, restored, `cmp`
  rc 0), the same test reads 0/48 at (3584, 3072, 2, 1, 1), the device's result. Cost: about
  70 s of the deviceless suite.

**`gated_delta_recurrent_f32`, the load-before-store form.**
- **Change.** `state_passes` now walks the state through `each_row`: four named row loads per
  group, then the four stores, then a scalar remainder.
- **ISA.** Both state loops of `gated_delta_recurrent_f32` and of `gdn_recurrent_f32` now issue
  four `global_load_b32` before their stores and consume them under `vmcnt(2)`/`(1)`/`(0)`,
  hipcc's `gated_delta` pattern in place of the one-in-flight `vmcnt(0)` chain. hipcc's `gdn`
  still serializes, so the Rust `gdn` should now beat it.
- **A first spelling spilled.** It used an array of four pointers, which landed in scratch
  (`scratch_store_b128`) with `flat` stores; named scalars fixed that.
- **Bits.** The same loads, multiplies and FMAs on the same values in the same order. A load
  issued before the preceding rows' stores reads the same value because the rows are distinct:
  the stride is ≥ 1, by the launchers' guard 1001. The device gate re-scores it.
- **The i32 counter, tried separately on top: rejected.** It added remainder copies (25 → 33
  loads, 19 → 27 stores), and its main loop already had the four loads in flight. The 64-bit
  counter stays.
- **Timing is owed to the next lease.** *(Paid 2026-11-19, the `79e218c` lease below.)*

**Device lease, 2026-11-19, commit `79e218c` (after the arena fix), two arms, rh-anine.** The
procedure was the `60ac104` lease's, with A then R run from one `setsid`-launched script. Both
witnesses CLEAN: 0 B witness files, GTT 17 MiB pre-arm. Host loadavg was 4.46 → 3.38 on A and
3.19 → 6.48 on R (32 cores). That is three to four times the `60ac104` lease's load, and is a
possible contributor to the two rows above 1.05 below. Neither arm was discarded.

| arm | binary | scripted verdict | rc | wall | logs `…/bit-identity/` |
|---|---|---|---|---|---|
| A | `bit_identity-efca1da5680a0280` (cargo's metadata hash, not a content hash); the script's freshness check passed at `79e218c` | GREEN: 23/23 BIT-IDENTICAL | 0 | 168 s | `20261119T133613-3151723` |
| R | `…/rk-l2-moe/plant/bit_identity-l2-plants`: `git archive 79e218c` plus the three `plants/*.patch`, built into `target/rk-l2-moe-plant` | GREEN (expect-red): exactly the five planted rows | 0 (binary 101) | 158 s | `20261119T133901-3157318` |

- **A.** The only NaN-payload-only words are on `moe_expert_range`: 1520. Every other row
  has 0.
- **What shows the down pass is now compared for real.**
  - On the device, only K3: the `moe_f4` plant reaches pass 2 and is red at both production
    shapes.
  - For GLM's VQ and int4 and for V4's fp4, it is inferred from the `Layout` fix and the CPU
    models; no plant reaches their down pass.
- **R — red cells by shape**, from the new tally:
  - `moe_expert_range_f4_situ` is **red at both K3 production shapes**: (3584, 3072, 2, 1, 1)
    49/64 and (3584, 3072, 0, 4, 1) 50/64. Also (96, 64, 0, 3, 1) 48/64, (160, 96, 1, 2, 1)
    50/64 and (32, 32, 0, 1, 1) 48/64. At most 16 of each shape's held seeds are NaN-`x` or edge
    seeds. The 1–2 edge seeds that went red are live because their live expert sits past the
    gate scales' planted NaN (see the dated note under "After the lease").
  - `moe_expert_range` and `moe_expert_range_i4`: every shape red, 56/64 at nrow 1 and 64/64
    at nrow 2. At nrow 1, `nan_some`'s NaN in `x` row 0 hides the SiLU on the 8 nasty seeds;
    row 1 of an nrow-2 cell has no NaN.
  - `gated_delta_recurrent_f32`: (96, 128) 64/64, (4, 32) 64/64, (3, 64) 64/64,
    (2, 2) 40/64, (1, 1024) 55/64 and (1, 1) 0/64.
  - `gdn_recurrent_f32`: 64/64 at every shape except (1, 1, 1, 1), 0/64.
  - At `d` = 1 the only `k` is the 100.0 edge, on every seed. `100 · rn(1/sqrt(10000 + 1e-6))`
    rounds to exactly 1.0f32, so the fused and unfused updates round identically.
  - Every unplanted row held, 0 red at every shape.

**TIME lines.** Same method as before: the minimum of three 200-launch batches, Rust ÷ hipcc.
These supersede the `60ac104` lease's MoE and recurrent rows. Every other row read
0.882–1.018 in A, except Phase 0's `attn_res` (in the table).

| row (shape) | hipcc µs | Rust µs | A ratio | R |
|---|---|---|---|---|
| `gated_delta_recurrent_f32` (96, 128) | 29.02 | 28.92 | **0.997** (was 1.647) | planted |
| `gdn_recurrent_f32` (16, 48, 128, 128) | 45.25 | 18.82 | **0.416** (was 0.999) | planted |
| `moe_expert_range` GLM (6144, 2048) nrow 1 | 112.06 | 113.06 | 1.009 | gate/up planted, down unchanged: 1.006 |
| `moe_expert_range` GLM nrow 2 | 121.29 | 125.75 | 1.037 | gate/up planted, down unchanged: 1.018 |
| `moe_expert_range_i4` GLM nrow 1 | 242.82 | 245.32 | 1.010 | gate/up planted, down unchanged: 1.008 |
| `moe_expert_range_i4` GLM nrow 2 | 313.75 | 335.85 | **1.070** | gate/up planted, down unchanged: 1.035 |
| `moe_expert_range_f4` V4 (4096, 2048), 1 expert | 55.27 | 56.82 | 1.028 | unplanted: 1.027 |
| `moe_expert_range_f4` V4, 6-expert batch | 766.64 | 787.40 | 1.027 | unplanted: 1.021 |
| `moe_expert_range_f4_situ` K3 (3584, 3072), 1 expert | 60.74 | 64.99 | **1.070** | planted |
| `moe_expert_range_f4_situ` K3, 4-expert batch | 707.37 | 707.06 | 1.000 | planted |
| `attn_res` (1, 9, 7168, 9) — Phase 0, owed | 53.62 | 57.43 | 1.071 | 1.013 (unplanted) |

- **The two delta rules.** The load-before-store groups closed `gated_delta`'s gap, and `gdn`'s
  predicted win landed. Both are single samples, since both were planted in R.
- **Rows above ~1.05.** Only the `79e218c` samples count here, because the `60ac104` MoE samples
  timed a NaN `h`.
  - **`moe_expert_range_i4` nrow 2: 1.070 in A.** R's planted-gate/up binary read 1.035 with
    the same down kernel. The row contains `moe_down_i4_r2`, the 16 → 12 waves kernel. With no
    A-vs-A floor, the crossing's cost is unmeasured.
  - **`moe_expert_range_f4_situ` single expert: 1.070 in A.** There is no second sample: it
    was planted in R.
- **Obligation 1, timing half.**
  - **V4: met.** The fp4 range reads 1.021–1.028 over two samples, with the down pass now real.
  - **K3: OPEN.** Its single expert reads 1.070 in the only valid sample.
