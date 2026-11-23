---
status: closed-shipped
scope: engine
verdict: COMPLETE 2026-11-20 — every HIP C++ kernel (66 launchers) is Rust on amdgcn-amd-amdhsa and hipcc is out of the build; the C++ survives only at tag archive/hip-kernels as the bit-identity reference. Witnessed on b5ca13a: 66/66 launcher rows byte-identical to the tagged hipcc, battery green bar the 30 inherited CodeScene files, determinism 1024/1024 under --arena-refresh, parity-glm 32/32, and GLM decode not slower than the pre-migration build under the pre-registered rule (3.031 vs 3.093 tok/s, identical ids and hit/miss counts). Follow-up (lease on afedb58): with GLM's JOIN 1 device_sync replaced by a zero-cost token, decode is still not slower than the pre-migration build and the attention excess is no longer measurable above the floor (+0.3 against a 0.53 ms same-binary spread, the two pairs without host load); the sync as the cause is post hoc, the attribution lease's pre-registered rule NOT MET. Two pre-existing MoE ordering items were owed (§5 Phase 2 follow-up): stage_batch's uploads closed 2026-11-20 by citation (blocking hipMemcpy waits for completion in HIP's docs and the CLR source; §5 stage_batch ordering, device tripwire h2d_ordering.rs); the trace-only poison fill is still owed. The body is the per-lane port log: open §1's hazards only when writing a new device kernel, and §5 Phase 2 for the closing evidence.
---

# Rust kernels — migration plan

> **COMPLETE 2026-11-20:** the hypothesis held; §5 Phase 2 is the closing evidence.

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
   > **CHANGED 2026-11-19:** the ordering moved from the asm's memory clobbers to two
   > `compiler_fence(SeqCst)`, same instructions (§5, Perf follow-up).
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
  > **DONE 2026-11-20 (Phase 2, §5):** `build.rs` builds only the device crate and links
  > `libamdhip64`; `crates/backend/kernels/` is deleted, and the C++ lives at the tag alone.

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
  > **PAID 2026-11-20 (Phase 2):** `tests/bit-identity.sh --census` is that judge, in the tree
  > (§5 Phase 2; `gate-red-proofs.md` §21).
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
- The barrier plant has no arm; its judge is owed (above). *(PAID 2026-11-20, §5 Phase 2.)*

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

### Phase 1 L4 (attn)

**Deviceless half (2026-11-19, branch `wave/rk-l4-attn`). Device arms are OWED to the GPU
lease.** *(PAID 2026-11-19 except W: `### Phase 1 L4 — device lease` below.)* `attn.hip` (950 lines, 6 kernels, 5 launchers + the scratch-size function) is ported
and deleted, and it is gone from `build.rs`'s hipcc list.

**Where it went:**
- **Kernels:** `crates/kernels/src/attn/` is split by shape, not by the C's order.
  - `mla.rs`: `mla_latent_attend`, `mla_attend_combine`.
  - `gqa.rs`: `gqa_attend` and `gqa_block_attend`, which are ONE body over a row window. The C
    spelled them twice, and `kernel-inventory.md`'s owed duplication note is closed.
  - `dense.rs`: `gather_attn_shared_kv`, `mha_attend`, sharing one `block_reduce`.
  - `mod.rs`: the wave score helper and the online-softmax `Running` state.
- **Argument blocks:** `rivoli_abi::kernarg::attn`. `MlaIo` is what the caller hands over;
  `MlaAttend` adds the split plan. `Gqa` is shared by `GqaRing`/`GqaBlock`.
- **Geometry:** `rivoli_abi::geom::attn`. `MLA_HB`/`TILE`/`ACC_REGS`/`MAX_SPLITS`,
  `GATHER_ATTN_*`, `GQA_*` and `MHA_THREADS` are constants both sides name, so no kernel reads
  blockDim.
- **Launchers:** `crates/backend/src/launch/attn.rs`. Guards, order and codes are the C's. The
  split plan and the scratch size are host functions.
- **`=>` rows:** in `hip_attn.rs`.

**Shapes this lane added for everyone** (commit `d6a6f1b`, `wave/rk-shared`):
`Kernel::launch_2d` covers the C's `dim3(x, y)` grids. attn is the only `.hip` file with
grid.y: the MLA split and the (head, row) grids. `LAUNCHED_BY` covers multi-kernel launchers.
The record is `gate-red-proofs.md` §19.

**Fused sites, read off hipcc's ISA:**
- The MLA half compiles with contraction on. hipcc fuses:
  - both score dots;
  - the accumulator's LEFT product, `fma(acc, corr, p·L)`;
  - `l·corr + p`;
  - the combine's two sums;
  - one site no source reading predicts. In `p = expf(s − m_new)`, the subtraction absorbs
    `s = scale·total` (`v_fma_f32 v, scale, total, −m_new`), while `m_new = fmaxf(m, s)`
    keeps the rounded `s`.
- From the gather kernel on, the C carries `contract(off)`, and that pragma holds to the end of
  the file, `mha_attend` included. Those four kernels' `v_fma`s are all `expf` range reduction
  or the IEEE divide, so the Rust carries no `fma` there.

**ISA census (deviceless):**
- Method: hipcc `-O3 --cuda-device-only -S` on the tag against `llvm-objdump -d` of the shipped
  code object, counted per kernel over each whole function (`.Lfunc_end`, not the first
  `s_endpgm`; a first count that stopped at an early exit undercounted hipcc).
- **FMA-class counts are identical in all six kernels:** 30/13/22/9/9/7.
- **Private segment is 0 B in all six.** A first build had 64 B in
  `gather_attn_shared_kv`: two `zip`-iterated loops over its `acc[4]` did not unroll. Indexed
  loops fixed it.
- VGPRs, hipcc → Rust: MLA 46 → 47, combine 15 → 15, gather 28 → 23, gqa 38 → 35, block
  37 → 35, mha 16 → 19. Every pair sits in the same 24-register occupancy granule, except
  gather, which drops a granule in Rust's favour.
- Static instruction count is 1.23–1.65× hipcc (`census.txt` TOTAL). `mha_attend`'s tree is unrolled against the constant
  `MHA_THREADS`, where hipcc loops over blockDim: 20 static `s_barrier` against 6. The TIME
  lines judge whether it costs anything.

**Bit-identity rows (5, one per launcher)**, `crates/engine/tests/bit_identity/attn{,_gqa}.rs`.
Inputs are in `attn_inputs.rs`, which is deviceless so the CPU plant models read the same
cells.
- 64 seeds per shape.
- Edge seeds carry `EDGES`. Every 8th seed plants ±inf/NaN in Q (and V for GQA), and e4m3's NaN
  codes in the MLA cache.
- Each row has guard cases that pin order.
- Branches reached:
  - **MLA**, at GLM's 64 heads:
    - nr 1/17/64 → 1 split;
    - 130 → 2 (ragged last);
    - 1000 → 13;
    - 2048 → 16, the ceiling;
    - 4100 → 16 (tps 17);
    - 2048 without scratch → the unsplit path at long context;
    - 128 heads at 2048 → 10 splits, the `by_grid` bound (tps 13);
    - 20 heads → an idle tail of waves, with sparse `rows` (duplicates possible);
    - kvl 256 → 8 of 16 registers.
    - The split partials are compared too: `InPlace` zero-filled scratch, so every written
      `acc`/`(m, l)` must match.
    - The C's scratch-size function is compared at 5 points.
  - **Gather:** V4's decode at topk 128 and 640, a 7-row prefill, d 100/1000/8, topk 2000
    (strided past the block) and 16118 (exactly at the C's LDS ceiling; the guard refuses
    16119). On nasty seeds query row 0 is
    fully masked, which must produce the C's NaN.
  - **GQA:** Glimmer's ring before filling, at the fill, past one wrap and deep in a later one;
    a global layer at rows 0 and 8191; a 4-row batch on a `win + tq − 1` ring; d 8/100/256.
  - **Block:** ctx 4096 (shipped), the golden's `q_offset` 0, clamps.
  - **MHA:** K3's 96 heads × 1024 behind the mask, kv 1, kv 8192 (the ceiling), unmasked odd
    widths. Nasty seeds give a fully masked row (NaN) with one NaN mask slot.
- **No aliased rows:** no production call of these launchers runs in place. There are four call
  sites (`gqa_block_attend` has none, unwired by decision), and every output is a distinct
  buffer.

**Red proofs.** One plant per family plus the 2-D axis, `tests/bit-identity/plants/attn-{mla,gqa,dense}.patch`,
each shown adversarial on CPU first (`attn_plants.rs`, `gate-red-proofs.md` §20):

| plant | expected red rows | CPU |
|---|---|---|
| `attn-mla`: right product fused | `mla_attend` | 35140/81920 acc values, 40/40 eligible seeds |
| `attn-gqa`: acc update fused | `gqa_attend`, `gqa_block_attend` | 4904/32768, 64/64 |
| `attn-dense`: score dot fused | `gather_attn_shared_kv`, `mha_attend` | gather 40214/65536 logits, 64/64; mha 22442/57600, 62/64 |
| `grid-y`: `blocks_y` passed as gridDimZ (§19's OWED slot) | the four 2-D rows | structural |

The MLA model's first plant was `p`'s fused site written unfused. It was **sub-threshold on
these inputs**: it moved only 41/64 seeds (`scratch/rk-l4-attn/plants-cpu.log`).
The exact-rounding subtraction only differs from the rounded one under cancellation. It is
recorded and was replaced by the hc_post defect class. The model also showed that 24/64 MLA
seeds are degenerate for any accumulator plant: the 8 nasty seeds are all NaN, and in the 16
scale-edge seeds one token's ×100 latent dominates. Its bar counts the other 40. Those classes
stay in the device rows for what they do test.

**Deviceless gates:**
- `cargo test --workspace --no-default-features` (`gates/test-nodef.log`): 505 passed, 1 failed (CodeScene, exactly the
  31 base files).
- Clippy is clean on both arms; jscpd is 0. The first kernel-covering build's clones were
  removed by restructuring, none exempted.
- Every new or touched file is CodeScene 10, except `hip_attn.rs` at 9.68, which is base.
- The census (`gates/census.log`, `--nocapture`) prints:
  - `20 Rust kernels: 18 rows (a gate row and a `=>` row each; 2 kernels rowed via
    LAUNCHED_BY), 1 argued exemption`;
  - `66 launchers: 66 with oracles`;
  - `53 with production callers, 13 classified uncalled`.

  Every log cited here is under `/var/cache/rivoli/scratch/rk-l4-attn/` (`gates/`, `isa/`).

**OWED to the lease** (PAID 2026-11-19 except W):
- arm A (18 rows);
- four expect-red arms on planted binaries (`grid-y` pays §19's `blocks_y` proof);
- TIME lines;
- the attention oracle suites, unmodified, in the battery — OWED (W, deferred to the
  integrated four-lane battery).

### Phase 1 L4 — device lease (2026-11-19, commit `731a047`)

Commit `731a047` on `d6a6f1b`. The coordinator's GO covered arms 1–5; W was deferred to the
integrated four-lane battery. Five arms ran in order through `tests/bit-identity.sh`, detached
with `setsid`, under the default lock. **No arm
was discarded.** Every witness file was 0 B, GTT was 17 MiB before each arm, and no arm logged a
fault. Host loadavg was 0.45 → 17.79 over arm A (presumed sibling builds, no process list taken)
and 1.08–3.05 in the later arms. Driver and logs: `/var/cache/rivoli/scratch/rk-l4-attn/gpu/`
(`arms.sh`, `arms.log`, `<arm>.out`), plus the log directories below.

| arm | binary | scripted verdict | rc | wall | logs `…/bit-identity/` |
|---|---|---|---|---|---|
| A | fresh, `bit_identity-efca1da5680a0280` | GREEN: 18/18 BIT-IDENTICAL | 0 | 427 s | `20261119T112806-2630525` |
| R-mla | `plant/bit_identity-attn-mla` | GREEN (expect-red): exactly `mla_attend` (551/704 cells) | 0 (binary 101) | 406 s | `20261119T113513-2713787` |
| R-gqa | `plant/bit_identity-attn-gqa` | GREEN (expect-red): exactly `gqa_attend` (544/640), `gqa_block_attend` (320/384) | 0 (101) | 393 s | `20261119T114159-2771059` |
| R-dense | `plant/bit_identity-attn-dense` | GREEN (expect-red): exactly `gather_attn_shared_kv` (187/448), `mha_attend` (209/320) | 0 (101) | 377 s | `20261119T114832-2804828` |
| R-grid | `plant/bit_identity-grid-y` | GREEN (expect-red): exactly the four 2-D rows (MLA 384/704, gather 128/448, gqa 192/640, block 320/384) | 0 (101) | 393 s | `20261119T115449-2818885` |

- **Arm A: every attention row is BIT-IDENTICAL.**
  - MLA: 704 cells, 337 567 744 words, 16 guards, partials included.
  - Gather: 448 cells; GQA: 640; block: 384; MHA: 320.
  - NaN-payload-only words: 0 for MLA and gather, 10 080 for GQA, 5 984 for block, 70 for
    MHA. All are on the nasty seeds, and all are NaN on both sides under the owner's rule.
  - The 13 Phase 0 rows stayed identical on the new `launch_raw`, the first device check of
    `d6a6f1b`'s 1-D path.
- **Every gather guard refused cleanly,** so the poisoned-index-buffer risk named before the
  lease did not arise.

**TIME, arm A (min of three 200-launch batches, hot shapes).** Every ratio above 1.05 is
flagged with ⚑.

| kernel (shape) | hipcc µs | Rust µs | ratio | ratio in the other arms where the row ran unplanted |
|---|---|---|---|---|
| ⚑ mla_attend (64 heads, nr 2048, 16 splits) | 424.22 | 509.66 | 1.201 | 1.191, 1.216 |
| ⚑ gather_attn_shared_kv (1, 64, 512, 640, 640) | 152.35 | 184.89 | 1.214 | 1.215, 1.216 |
| ⚑ gqa_attend (decode, pos 4095, ring 2048) | 2598.50 | 2969.34 | 1.143 | 1.147, 1.141 |
| ⚑ gqa_block_attend (ctx 4096, block 16) | 9132.52 | 9798.95 | 1.073 | 1.067, 1.061 |
| mha_attend (96 heads × 1024) | 2333.48 | 634.41 | **0.272** | 0.272, 0.278, 0.267 |

- **The attention ratios repeat across arms within ±0.03.** They are not this lease's noise.
- That cannot be said of the three gated combines, the same code Phase 0 timed at 0.96–1.17
  across leases. They read 1.15–1.17 here (swiglu 1.167, clamped 1.174, situ 1.148: Rust
  2.46–2.51 µs vs hipcc 2.09–2.19 µs), but 0.97–1.00 in the four plant arms, except R-mla's
  `situ_glu_f32` at 1.201. They remain the unexplained swing recorded at the merge-review
  lease.
- The rest of Phase 0, in arm A, is within 0.86–1.04: `rmsnorm_weightless_batch` 0.863,
  `attn_res` 1.035, `hc_pre` 1.000, `rope_*` 1.00/1.02. `attn_res` reads 1.03–1.07 across
  the arms (1.071 in R-gqa), still the 6% Phase 0 owed.
- **`mha_attend` runs at 0.27× hipcc**, the reverse of the risk named before the lease. The
  Rust tree is unrolled against the constant `MHA_THREADS`, and the kernel never reads
  blockDim. Why that buys 3.7× is not isolated.

**ISA evidence for the ⚑ rows (deviceless, read after the lease):**
- **MLA: the C's `int` loops are `nsw`, `gpu::for_i32`'s are not.** `i += stride` wraps in
  Rust, so LLVM cannot turn the score dots' `qa[i]` into a pointer increment. Every iteration
  re-extends and re-shifts the index: `v_ashrrev_i32` + `v_lshlrev_b64` + a 64-bit add
  (`isa/rs_mla_latent_attend.s`), where hipcc adds `0x80` to a pointer
  (`isa/mla_latent_attend.s`, `.LBB0_23`).
  - Experiment: a scratch copy of `crates/kernels` with `for_i32`'s increment changed to
    `unchecked_add` (`scratch/rk-l4-attn/exp-nsw/`).
  - Result (counted by `census.sh`'s rule, `exp-nsw/counts.txt`): MLA drops from 1249 to
    1107 static instructions (hipcc 979), and both dot loops become the same instructions as
    hipcc's, scheduled slightly differently.
  - **Untimed.** It is a shared Phase 0 helper (every `for_i32` kernel), so the change and
    its device check belong to the coordinator, not this lane.
  - The same experiment leaves gather (918 → 914) and gqa (915 → 915) essentially unmoved, so
    it is not their cause.
- **Gather and GQA: no cause isolated.** The logits loops match hipcc's shape (gather: ×8
  unrolled, `global_load_b128`).
  - Gather's values pass carries `u64` loop-counter bookkeeping hipcc does not have
    (`for j in 0..topk as usize` against the C's `int j`), plus the same per-column exec-mask
    branches.
  - GQA's static size is 1.33× hipcc, with occupancy equal or better.
  - **OWED:** these are attention's decode-path kernels.

### Phase 1 L3 (mla, kvcompress, blockindex, indexer)

**indexer + blockindex, deviceless half (2026-11-19).**

*Ported.* Ten kernels in two families: the eight `indexer.hip` kernels and `blockindex.hip`'s
`act_quant_f4_rotated` and `index_score_blocks`. Both `.hip` files are deleted.

*FMA sites.* Each was read off hipcc's ISA:
- `+= a * b` dot accumulators are `v_fmac`;
- `layernorm`'s variance is `fma(-mean, mean, Σx²/n)` and its output
  `fma((x − mean)·inv, w, b)`;
- neither block scorer has any (both are `contract(off)`).

*ISA census.* FMA, div, rcp and sqrt counts match hipcc on nine of the ten kernels. The
remaining count differences are unroll (constant-trip trees and scans unrolled, the scorers'
inner loop unrolled 4×). `act_quant_f4_rotated` has fewer fma/div/cvt ops in Rust because hipcc
unrolled its 32-element fp4 round-trip loop 2× (`%div.i`/`%div.i.1` in its IR). No kernel uses
scratch.

*Run-time blocks.* `index_score`, `index_head_route` and `index_pool_norm_f32` take their
launcher-sized block in `kernarg::Blocked`, never from the dispatch packet.

*Shared operand set.* `ScoreBufs` is now `rivoli_abi`'s, one definition embedded in the
scorers' `ScoreBlocks`. Both scorers became `=>` rows. **The `hip.rs` exemption did NOT retire,
measured:** with its markers deleted, jscpd found 2 clones:
- the two rows' parameter lists;
- `memcpy_dtod_async` against `hiprt.rs`, hidden there since before the migration.

So the count stays 9.

*Harness.* Integer outputs (top-k rows, the bf16 cache) are compared byte for byte
(`Out::Exact`; *2026-11-19, Phase 1 integration: now `Out::PoisonBits`/`Out::InPlaceBits`, the
one integer-output pair three lanes' variants were unified into*). The top-k scores are tie-heavy by construction (`inputs::topk_scores`).

*Plants.* Two launcher-row plants, each with a CPU proof, recorded in `gate-red-proofs.md` §18.

**kvcompress, deviceless half (2026-11-19).**

*Ported.* `gemm_bf16`, `kv_compress_deposit`, `kv_compress_prefill`, `kv_compress_decode`;
`kvcompress.hip` deleted. `CompGeom`/`CompFinish` are no longer mirrors: they ride by value
inside the kernels' argument blocks.

*The deposit's grid stride.* It read `gridDim.x · blockDim.x`, so it now arrives as
`kernarg::Strided::threads`, set by the launcher from the grid it launches.

*A defect introduced by an ISA misreading, and caught by review on the IR.* The first draft
fused the RoPE `hi` correctly: hipcc fuses the LEFT product of `a·s + b·c`, `fma(a, s, b·c)`.
That is `%add84 = fadd(%mul82 = s·a, %mul83 = c·b)` in the IR, and in the ISA
`v_mul c·b; v_fmac += s·a`, where `v[12:13]` is `seg`, the pair the store writes back.

A census follow-up then swapped the register roles, "fixed" the kernel to the right product,
and wrote a correction to `kernels/lib.rs` saying the left-product rule does not hold. The
correctness reviewer re-derived both sites from the IR. Phase 0's `rope` is written
`b·cs + a·sn`, so its fused `b·cs` is also the left product. Everything was reverted before
commit; the rule stands.

Two lessons:
- Read register roles off the loads and stores, not off which product "looks" like the pair.
- The CPU model shows the gate would have seen this defect on only 10/384 prefill cells
  (`gate-red-proofs.md` §18, Phase 1 L3).

*The masked-`ape` `+ 0.0`.* The decode pool's add survives (`-0.0 + 0.0 = +0.0`); it is
present in the Rust ISA.

*ISA census.* Identical to hipcc for `gemm_bf16` and the deposit. The pooling kernels differ
by hipcc's 3–4× unroll of the entry loops.

*Device arms.* OWED.

**mla, deviceless half (2026-11-19).**

*Ported.* Nine kernels behind seven launchers:
- `mla_absorb_fp8{,_r2}` and `mla_value_fp8{,_r2}`, where each launcher picks a kernel by token
  rows;
- `act_quant_f8_prefix`, `rmsnorm_batch`, `qk_norm`, `rope_adjacent` and `gemv_fp8_bf16`.

`mla.hip` is deleted, and `absorb_nquad` moved from the prelude to `rivoli_abi::geom`, closing
the grid tie its note had left open.

*A file-scope pragma decided the bits.* `mla.hip`'s `#pragma clang fp contract(off)` sits before
`act_quant_f8_prefix` at FILE scope, so every kernel after it is unfused in hipcc's IR. The first
draft fused `rmsnorm_batch`'s `Σx²` and `rope_adjacent`'s rotation, by the habit of every other
norm and RoPE in the tree. The ISA census caught both before any device run: hipcc's
`rope_adjacent` has 4 `v_mul`, 1 `v_add`, 1 `v_sub` and no fma. `gemv_fp8_bf16`'s dot stays
fused because `dot_fp8.hpp` is included above the pragma.

*The harness compared nothing on one row, and review found it.* `fp8_words` first drew every
byte, the e4m3 NaN codes included. At k = 4096 essentially every GEMV output was NaN on both
sides, NaN compares equal as a class, and the hot shape examined zero finite words. The draws
now remap the NaN codes to their finite neighbours.

*Two more measured gaps, fixed.*
- Guard cases now assert the exact code they claim (`Tally::guard_code`), with the
  non-1001 orderings included.
- The ragged GEMV shape no longer makes misaligned dword loads.

*Register spill, found and fixed.* The first `mla_absorb_fp8` spilled 32 B (R=1) and 160 B (R=2)
to scratch where hipcc spills none: iterator-indexed accumulator arrays. Constant-bound index
loops removed it.

*ISA census.* After the fixes, every float op class matches hipcc for the value pair, the norms,
the RoPE and the GEMV. The absorb differs only by unroll and 64-bit index arithmetic.

*Census.* The two `_r2` kernels are rowed through `LAUNCHED_BY` (main `d6a6f1b`, merged
2026-11-19). Each launcher names both statics in its own body (`run(&MLA_*_R2, …)` beside
`run(&MLA_*, …)`), and its row sweeps `nrow ∈ {1,2}`. The census is green, printing "37 Rust
kernels: 34 with a gate row and a `=>` row, 1 argued exemption" (`cargo test -p rivoli
--no-default-features --test kernel_coverage -- --nocapture`, 2026-11-19).

*Device arms.* OWED.

**The L3 device lease (2026-11-19, commit `bf43d2e`).**

Five arms ran in order, once each, all through `tests/bit-identity.sh`.

**The witness was clean on every arm:** empty KFD witness file, GTT 17 MiB pre-arm. Host
loadavg ran between 0.6 and 3.8 on 32 cores.

Logs: `/var/cache/rivoli/scratch/rk-l3-mla/gpu/lease-20261119T120910/` (`progress`, per-arm
`.out`/`.err`) and `/var/cache/rivoli/scratch/bit-identity/20261119T12{0910,1320,1731,2140,2551}-*`.

| arm | scripted verdict | rc | wall |
|---|---|---|---|
| A (34 rows: the 13 Phase 0 rows + L3's 21) | GREEN: 34 of 34 BIT-IDENTICAL, NaN-payload-only 0 on every row | 0 | 250 s |
| plant `indexer` (`--expect-red=index_topk`) | GREEN (expect-red): 126 of 448 cells red, all partial selections | 0 | 251 s |
| plant `blockindex` (`--expect-red=index_score_blocks`) | GREEN (expect-red): 203/256 cells red | 0 | 249 s |
| plant `kvcompress` (`--expect-red=gemm_bf16`) | GREEN (expect-red): 266/320 cells red | 0 | 251 s |
| plant `mla` (`--expect-red=mla_absorb_fp8`) | GREEN (expect-red): 256/384 cells red | 0 | 249 s |

**Every plant's device red count equals its CPU model's count**: 126, 203, 266, 256. The CPU
models drew the gate's own inputs (`gate-red-proofs.md` §18, Phase 1 L3).

**TIME, L3's rows** (arm A; min of three 200-launch batches, hot shape, null stream). This is
one arm with no A-vs-A noise floor measured, so ratios within a few percent of 1 do not rank:

| row | hipcc µs | Rust µs | ratio |
|---|---|---|---|
| layernorm 128 | 3.05 | 2.68 | 0.881 |
| index_append (5, 128) | 2.02 | 2.03 | 1.001 |
| index_score (4096, 64, 64, 128) | 50.27 | 50.86 | 1.012 |
| index_pool_push (1, 128) | 2.06 | 2.03 | 0.989 |
| index_head_route (4, 64, 128) | 4.28 | 4.26 | 0.995 |
| index_topk (4096, 2048) | 44.88 | 46.12 | 1.028 |
| index_pool_norm_f32 (512, 4, 128) | 4.81 | 4.48 | 0.932 |
| index_score_blocks_f32 [1, 512, 64, 128] | 627.98 | 114.66 | 0.183 |
| **act_quant_f4_rotated (64, 128)** | 4.98 | 7.31 | **1.469** |
| index_score_blocks [1, 512, 64, 128] | 627.92 | 115.78 | 0.184 |
| gemm_bf16 [1, 1024, 4096] | 29.77 | 30.98 | 1.041 |
| **kv_compress_deposit (decode step)** | 2.14 | 2.27 | **1.060** |
| kv_compress_prefill (ratio 4, 512 blocks) | 88.06 | 89.87 | 1.020 |
| **kv_compress_decode (ratio 4, d 512)** | 11.72 | 14.31 | **1.222** |
| mla_absorb_fp8 (GLM-class, 1 row) | 27.65 | 25.73 | 0.931 |
| **mla_value_fp8 (GLM-class, 1 row)** | 27.61 | 29.12 | **1.055** |
| act_quant_f8_prefix (1, 4096) | 54.67 | 53.88 | 0.985 |
| rmsnorm_batch (1, 1024) | 3.59 | 3.32 | 0.925 |
| qk_norm (64, 512) | 3.36 | 3.00 | 0.892 |
| rope_adjacent (64 heads, 512) | 2.82 | 2.57 | 0.911 |
| gemv_fp8_bf16 (V4 `wo_a` slice) | 13.05 | 13.27 | 1.017 |

**Four rows exceed 1.05, reported and not fixed. The causes below are read off the ISA; none is
isolated.**

*L4's suspect.* L4 suspected the shared `gpu::for_i32` helper: its increment lacks the C's `nsw`,
which shows as a sign-extend, a shift and a 64-bit add per iteration. Counted with
`llvm-objdump --disassemble-symbols=<k>` on the embedded object against hipcc's `.s` in
`/var/cache/rivoli/scratch/rk-l3-mla/` (`v_ashrrev_i32|v_lshlrev_b64|v_lshl_add_u64`):
- Three rows carry FEWER of these instructions than hipcc: `kv_compress_decode` 12 vs 18,
  `mla_value_fp8` 8 vs 16, `kv_compress_deposit` 4 vs 6.
- **`act_quant_f4_rotated` carries the pattern**, 4 vs 0, found by the correctness review of this
  record. Its two global loops over `row[i]` (load into LDS, store back) each sign-extend `i`
  and shift it in 64 bits per iteration, where hipcc increments a pointer.
- At the timed shape, though, those loops run ONCE per thread (d = 128 over 128 threads). So
  the pattern is present but cannot explain 2.3 µs there. It would weigh at wider rows, and the
  shared helper is L4's to change.

*What the ISA does show* (op counts from `bin/census.sh`):
- **`act_quant_f4_rotated`, 1.469.** At d = 128 only 4 threads per block do the fp4 work, each
  a serial 32-element round trip with an IEEE divide per element. hipcc unrolls that loop 2×
  (`%div.i`/`%div.i.1` in its IR); Rust leaves it rolled. The inferred cause is the dependency
  chain, about twice as long. Fix candidate: a manual 2× unroll. It should not change bits,
  since every element is independent; a gate re-run is owed either way.
- **`kv_compress_decode`, 1.222.** This is a single-block kernel, so it is latency-bound, and
  its pooling loops run over a run-time `ents`. hipcc unrolls the entry loops about 4× (census:
  exp 8 vs 2, max 27 vs 2); Rust keeps them rolled. Fix candidate: the same manual unroll.
- **`kv_compress_deposit`, 1.060, and `mla_value_fp8`, 1.055.** These are within ~0.1–1.5 µs on
  launch-sized work: the deposit is 2.1 µs, and the value pair's float census is identical to
  hipcc's. They sit in the same band as Phase 0's `swiglu` (1.050). Not investigated further.

**The block scorers run 5.5× faster than hipcc.** hipcc's `contract(off)` dot loop runs rolled
with 64-bit index arithmetic per iteration. Rust unrolls it 4× with wider loads (census: gload
7 vs 3).

Also outside this lane, for the record: Phase 0's `swiglu_clamped_bf16` (1.135) and
`situ_glu_f32` (1.137) exceeded 1.05 on this lease.

### Phase 1 L1 (linalg, vmm, headtail, fwd)

**Deviceless half, 2026-11-19 (branch `wave/rk-l1-linalg`, base `933554c`).** All four files
are deleted and out of `build.rs`'s hipcc list. Device arms: OWED to the GPU lease.
> **2026-11-19:** the bit-identity arms are paid on the device (the lease below); the
> oracle-suite battery W, which also exercises the vmm FFI, is still OWED.

- **What moved.** 23 kernels became Rust: `crates/kernels/src/{linalg,fwd,headtail}.rs`
  (`gemv_fp8`/`_r2`, `gemv_fp8_splitk`/`_r2`, `gemv_i8`/`_r2`, `gemv_f32`/`_r2`, `gemv_i4`,
  `gemv_vq`, `vq_encode`; `embed_i8_row`, `append_kv`, `gather_rope`, `vadd`, `sigmoid_gate`,
  `logit_softcap`, `argmax` (the C's `argmax_reduce`, renamed for its launcher), `flag_nonfinite`,
  `hash_rows`; `embed_bf16_row_bcast`, `hc_head_collapse_gate`/`_blend`). Their 17 launchers are
  `launch/{linalg,fwd,headtail}.rs` behind `=>` rows. vmm.hip was host-only: `rivoli_vmm_alloc`/
  `_free` are `hiprt.rs` FFI now, returning a `VmmMapping` instead of filling three
  out-pointers. `hash_rows` was hand-written in `hip.rs` for an 800-line cap `hip_linalg.rs` no
  longer presses; it is a `launchers!` row, and the wrapper's `i32` overflow bail is a guard
  code (1002, now also for a `stride` past `i32` and an `n + stride − 1` that overflows — the C
  wrapped both silently).
- **ABI.** The GEMVs share ONE generic argument block, `kernarg::Gemv<W>`, the weight operand
  `W` the only field the formats differ in (`Scaled`, `*const f32`, `Vq`); `GemvFp8` adds
  `block`. Block sizes: `GEMV_BLOCK = ROWS_PER_BLOCK · WAVE` (not `geom::ROW_BLOCK`, the norms'
  tree, which is 256 by coincidence), `ARGMAX_BLOCK`, `HASH_BLOCK`, `HC_HEAD_GATE_BLOCK = WAVE`,
  `ELEMENTWISE_BLOCK` for the rest. `hash_rows` read `gridDim·blockDim` for its grid stride; the
  launcher passes it (0c's rule). `append_kv` launches `kvl` threads and reads only `threadIdx`.
- **Fused sites** were read from the tag's ISA (`hipcc -O3 --cuda-device-only -S`) and are
  listed with their instructions in each kernel module's header. The one worth repeating:
  `gemv_i8`'s dword step is `acc + quad` with the quad fused and the accumulate a PLAIN
  `v_add`, because no scale wraps it (fp8's and int4's are `fma(s, quad, acc)`).
- **ISA census, hipcc vs Rust, all 23 kernels** (`census.py`-style count of FMA-family ops incl.
  both VOPD halves, `v_mul_f32`, `v_add_f32`, `v_sub_f32`): **identical in 22**; the 23rd,
  `hc_head_collapse_blend`, has 5 FMAs to hipcc's 1 because LLVM unrolled its runtime `hc` loop
  by four — schedule only, the fold stays ascending. Scratch **0 B in all 23** (`llvm-readelf
  --notes`); VGPRs within one granule of hipcc's everywhere (max 35, `gemv_i4`, hipcc 32).
  **The scratch check caught one:** threading `gemv_i8`'s accumulators as `&mut [f32; R]` and
  stepping them with `iter_mut` put 8 B (R=1) / 12 B (R=2) in scratch and moved its FP counts;
  passing them by value through `from_fn` restored 0 B and the identical census. Other static
  differences, for the timing arm to price: `argmax`/`hash_rows` unroll their 256-leaf tree
  (`s_barrier` 2 → 9, the C's `blockDim.x/2` loop became a constant); `hash_rows`' system-scope
  `fetch_xor` lowers to a `global_atomic_cmpswap_b64` loop where hipcc's agent-scope `atomicXor`
  is one `global_atomic_xor_b64` (one per block, ≤ 1024 blocks; bits are the same either way);
  `embed_bf16_row_bcast` unrolls its `hc` stores. (Priced 2026-11-19: see the device lease below.)
- **Gate rows.** 17 rows (`bit_identity/{linalg,fwd,headtail}.rs`), ≥ 64 seeds, the engine's
  `i_dim`s at a few dozen rows plus a timed hot shape that fills the machine (a GEMV row's bits
  depend on `i_dim`, never `o_dim`). Every tail is reached: fp8 `block < 4` and the one-aligned-
  row `i_dim % 4` tail, the split-K route and its 4096 threshold, int8's scalar tail and
  misaligned-row fallback, int4's unaligned rows, `append_kv` at `pos` 131071, argmax ties /
  all-NaN / all-`-inf` / `+inf`, the hash past its 1024-block cap. NaN codes and NaN scales are
  PLANTED on one seed in eight (a random fp8 code is NaN 2 in 256, which would NaN every 6144-long
  dot and leave the NaN-class compare nothing to see). New harness pieces: `Out::PoisonBits`/
  `ZeroBits` (*2026-11-19, integration: `ZeroBits(n)` is now `InPlaceBits(vec![0.0; n])`*) — integer outputs compared bit for bit, never as a NaN class — and `guards()`. Slab
  outputs (`append_kv`) are sized to row `pos` exactly, the kernel handed the base `pos` rows
  before it, so the per-side poison sees every element.
- **Red proofs, CPU first** (`bit_identity/linalg_plants.rs`, deviceless, patches in
  `tests/bit-identity/plants/{linalg,fwd,headtail}.patch`). *2026-11-19 (integration): the file
  was `plant_models.rs` on the lane branch; renamed because L3's file of that name landed too.*

  | plant | expect red | CPU (cells moved) |
  |---|---|---|
  | `linalg`: `lane_cols` unfused | `gemv_f32`, `gemv_i8` | 64/64 (`gemv_f32` 160×6144); 64/64 (`gemv_i8` scalar-tail rows) |
  | `fwd`: `sigmoid_gate` as `x / (1+e)` | `sigmoid_gate` | 64/64 at n = 257, 2048, 6144 |
  | `headtail`: the gate's `lane_dot` unfused | `hc_head_collapse` | 64/64 (seed over s = 1 and 3) |

  **Two first plants were sub-threshold, and the CPU model refused them:** the gate's mix
  `fma(m, scale, base)` split moved 18/64 cells (the sigmoid swallows it), and `gemv_i8`'s tail
  counted on one shape's misaligned rows alone moved 40/64. The `fwd` alternative
  `f2e4m3`'s `fma(a, 512, 0.5)` split was not tried: `a·512` is exact, so it is a no-op by
  construction (§1's lesson, caught on paper).
- **Census.** Before main's `LAUNCHED_BY` table (d6a6f1b) the gate-row census was RED on
  exactly the seven kernels no same-named launcher runs — `gemv_fp8_r2`, `gemv_fp8_splitk`/`_r2`,
  `gemv_i8_r2`, `gemv_f32_r2`, `hc_head_collapse_gate`/`_blend` — and those are its seven rows
  now. The table's rule (the launcher's own body names each static) replaced the launchers'
  `[Kernel; 2]` arrays with one static per instantiation. After the merge: 5/5 censuses green.
- **Deviceless gates.** jscpd 0 clones (≈ 30 found on the way, removed by restructuring, none
  exempted — including one against `rivoli_core::num`'s e4m3 oracle); clippy both arms clean;
  CodeScene 10.0 on every file touched except `hip_attn.rs` (9.68 at the base, two rows edited);
  `cargo test --workspace --no-default-features --no-fail-fast` on the merged tree: 505 passed,
  1 failed — the CodeScene test (the 31 base files); docs, hook_guard, kernel_coverage,
  line_limit and matrix each confirmed RUN and green.

**Device lease, 2026-11-19 (commit `3ab7b71`, the merged tree).** Arms A, Rlinalg, Rfwd and
Rheadtail ran once each, in that order, detached (`setsid`, own PGID), through
`tests/bit-identity.sh`. The plant binaries were `3ab7b71` plus
`tests/bit-identity/plants/<family>.patch`, built outside the lock into a separate target dir
(`/var/cache/rivoli/scratch/rk-l1-linalg/plant/build.sh`), and the tree was restored, checked
by `git diff`. **Witness clean on all four:** every witness file 0 B, GTT 17 MiB before each arm
(as `gpu-witness.sh` prints it), host loadavg 0.6–3.9 on 32 cores. Logs:
`/var/cache/rivoli/scratch/bit-identity/20261119T123500-2927154` (A), `…/20261119T123727-2934726`
(Rlinalg), `…/20261119T123956-2943625` (Rfwd), `…/20261119T124219-2953372` (Rheadtail); per-arm
stdout and `arms.log` in `/var/cache/rivoli/scratch/rk-l1-linalg/gpu/`.

| arm | scripted verdict | rc (script / test binary; 101 expected under `--expect-red`) | wall |
|---|---|---|---|
| A | GREEN: 30 of 30 rows BIT-IDENTICAL, NaN-payload-only 0 on every row | 0 / 0 | 147 s |
| Rlinalg | GREEN (expect-red): exactly `gemv_f32` (196/320 cells, 30346 words) and `gemv_i8` (172/448 cells, 669 words) | 0 / 101 | 149 s |
| Rfwd | GREEN (expect-red): exactly `sigmoid_gate` (256/320 cells, 405603 words) | 0 / 101 | 143 s |
| Rheadtail | GREEN (expect-red): exactly `hc_head_collapse` (167/320 cells, 530 words) | 0 / 101 | 141 s |

The 30 rows are Phase 0's 13 plus L1's 17. The 17 reach all 23 L1 kernels: the `_r2` and split-K
instantiations through the R = 2 and `i_dim >= 4096` shapes, and both `hc_head_collapse` phases
through its one row. The Phase 0 rows stayed identical through the `gate_map → elementwise`
refactor. **Still OWED:** the oracle-suite battery (W, script prepared at `…/gpu/W.sh`, dropped
from this lease by the coordinator to run once on all four lanes' integrated result). The vmm
host port (`rivoli_vmm_alloc`/`_free`) has no device evidence until then, because no
bit-identity row reaches it.

**Timing.** Each plant arm leaves every L1 kernel
unmodified except its planted one, so the four arms are same-code replicates — the A-vs-A
control Phase 0 said it lacked. The range below is over the arms where the kernel is
unmodified: 3 arms for `gemv_f32`, `gemv_i8`, `sigmoid_gate` and `hc_head_collapse`, 4 for the
rest. **No parity band was pre-registered.** The coordinator's brief flags > 1.05. The measured
cross-arm spread of unmodified ~2 µs rows in this lease is 0.94–1.17, which sits above that line,
so a verdict on a ~2 µs row below ~1.15 is inside noise.

| kernel (hot shape) | hipcc µs (A) | Rust µs (A) | ratio range | reading |
|---|---|---|---|---|
| `gemv_fp8` (2048, 6144, 128, 1) | 31.69 | 32.80 | 1.029–1.168 | 1.029, 1.035 in A/Rlinalg; 1.070 Rheadtail, 1.168 Rfwd: > 1.05 in 2 of 4, no cause sought |
| `gemv_fp8` split-K (1024, 16384, 128, 1) | 36.31 | 38.49 | 1.017–1.087 | **> 1.05 in 3 of 4** |
| `gemv_i8` (2048, 6144, 1) | 28.34 | 30.37 | 1.058–1.076 | **slow** |
| `gemv_i4` (2048, 6144, 1) | 17.36 | 17.13 | 0.947–0.999 | parity |
| `gemv_vq` (2048, 6144) | 32.87 | 33.53 | 1.015–1.020 | parity |
| `gemv_f32` (256, 7168, 1) | 42.45 | 46.29 | 1.090–1.104 | **slow** |
| `vq_encode` 8192 | 117.30 | 188.01 | 1.603–1.622 | **slow** |
| `embed_bf16_row_bcast` (5, 4096, 4) | 2.13 | 2.13 | 1.000–1.027 | parity |
| `hc_head_collapse` (1, 4, 4096) | 239.77 | 236.73 | 0.987–0.999 | parity |
| `embed_i8_row` (5, 6144) | 2.04 | 2.03 | 0.994–1.147 | within the 2 µs spread |
| `append_kv` (512, 64, 777) | 2.60 | 2.65 | 1.003–1.038 | within the 2 µs spread |
| `gather_rope` (64, 192, 128, 64) | 2.04 | 2.00 | 0.982–1.118 | within the 2 µs spread |
| `vadd` 6144 | 2.00 | 2.01 | 0.982–1.141 | within the 2 µs spread |
| `sigmoid_gate` 6144 | 2.03 | 1.99 | 0.980–0.987 | parity |
| `logit_softcap` 202048 | 3.95 | 3.82 | 0.942–0.969 | parity |
| `argmax` 151552 | 86.35 | 85.69 | 0.992–1.007 | parity in all 4 |
| `flag_nonfinite` 6144 | 2.02 | 2.01 | 0.986–1.133 | within the 2 µs spread |
| `hash_rows` (6144, 1) | 2.79 | 5.27 | 1.808–1.933 | **slow** |

In Rfwd, the L1 ~2 µs rows `embed_i8_row`, `gather_rope`, `vadd` and `flag_nonfinite` read
1.118–1.147. In the other three arms they read 0.98–1.04. The Phase 0 gate kernel `swiglu` read
0.981–1.171 across the four arms, and `attn_res` read 1.046–1.082 (still owed, as before; *paid 2026-11-19, 0.995 / 1.000 on the
integration lease, Phase 1 integration Step 4*).
This is the same unexplained ~2 µs swing as Phase 0's merge-review lease, now with replicates.

**Slow rows, read from the ISA** (`llvm-objdump` of the shipped object against the tag's
`--offload-device-only` object, loop bodies compared). Two compiler-level classes were named by
the L3/L4 leases. **(a)** `gpu::for_i32`'s wrapping `i32` increment carries no `nsw`, so each
iteration sign-extends and shifts the index into a 64-bit address, where hipcc strength-reduces
to pointer increments. **(b)** rustc's LLVM leaves a runtime-count loop rolled that hipcc
unrolls (or the reverse). **Each cause below is the only loop-body difference found. None was
ablated, so each is a candidate, not a priced cause.**
- **`gemv_f32` — (a).** Its lane loop is `for_i32`. Each Rust iteration does `v_ashrrev_i32` +
  `v_lshlrev_b64` + two 64-bit adds. hipcc's loop increments two pointers by `0x80`. Neither side
  unrolls, so not (b).
- **`gemv_i8` — (a), twice.** The scalar tail is `lane_cols` → `for_i32`, the same pattern. The
  dword loop is this file's own `while base + step <= i_dim`, whose
  `col = base as usize + lane·4` re-derives the `x` address with a `v_lshlrev_b64` per iteration,
  where hipcc increments its pointers by `0x80`/`0x200`. That is the same missing-`nsw` class,
  outside the shared helper.
- **`vq_encode` — neither class; both sides unroll ×4.** `bk` is a `usize` where the C's is
  `int`, so every argmin update is two `v_cndmask_b32_e64` instead of one. The `0..VQ_K` usize
  range keeps `Range::next`'s guarded increment (`k + (k < 0x1000)`, a `v_cmp_gt_u64`) in every
  unrolled copy. That makes each copy's codebook address depend on the previous compare, which
  plausibly blocks merging the four rows into hipcc's one `s_load_b512`; Rust issues four
  `s_load_b128` and four `s_load_b32` instead. Fix candidate: an `i32` counter and an `i32` `bk`,
  as the C has (applied — see the 2026-11-19 note below).
- **`hash_rows` — neither class: atomic scope.** Rust's system-scope `AtomicU64::fetch_xor`
  lowers to a `global_load_b64` + `global_atomic_cmpswap_b64` retry loop, and 24 blocks contend for
  one address. hipcc's agent-scope `atomicXor` is one `global_atomic_xor_b64`. The unrolled
  256-leaf tree (`s_barrier` 2 → 9) is the other difference, and these numbers cannot separate the
  two. `argmax` carries the same tree change at parity, which points to the atomic. The cost is
  ~2.5 µs over hipcc per call, live only under `--divergence-log`. A fix needs an agent-scope XOR,
  which `core::sync::atomic` does not expose.
- **`gemv_fp8` split-K — one difference, unpriced.** The main loop is the prelude's `usize`
  `for_strided`, with no (a) pattern. hipcc's own loop here carries it (three `v_ashrrev_i32`,
  two `v_lshlrev_b64` per iteration) and is still faster, so (a) alone is not a sufficient cost
  model. The loads match. The combine reads the eight partials as `ds_load_2addr_b32` pairs where
  hipcc uses two `ds_load_b128`. Whether that is the 2–9% is unmeasured.

> **ISA-MATCHED 2026-11-19 (deviceless; bit-identity and timing OWED to the combined branch's
> device run, still to come):** the two slow rows whose causes were in this lane's own code.
> Whether these were the 1.06× and 1.61× is unmeasured until that run.
> - **`vq_encode`:** an `i32` range and an `i32` `bk`, as the C has. The loop now has hipcc's
>   structure: one `s_load_b512` (four codebook rows) and one `s_load_b128` (four norms) per
>   unrolled step, one `v_cndmask_b32_e64` per argmin update, no `v_cmp_gt_u64`, and exit on
>   `s_cmpk_lg_i32 s26, 0x1003`. An `i32` `for` range and a hand-written `i32` counter compile
>   to the same merged loop (both tried), so the `usize` type was the blocker.
> - **`gemv_i8`'s dword loop:** a `usize` column stepped beside the `int` bound. The loop no
>   longer has the per-iteration index sign-extend (`v_ashrrev_i32 …, 31`) or `v_lshlrev_b64`.
>   The remaining `v_ashrrev_i32 …, 24` is the top-byte decode, and hipcc has it too. `x` is a
>   pointer bumped by `0x200`, as in hipcc, and the weight offset is a scalar bumped by `0x80`. A
>   first attempt (an array of advanced pointers) lost `x`'s global address space
>   (`flat_load_b128`); the ISA read caught that, and it was rejected. The scalar tail is still
>   `for_i32` (cause (a)), left for the shared helper's change at integration.
>
> Bits should not move. That is argued, not gated: both are pure index and type changes over the
> same addresses, and the FP-op census still matches hipcc for `vq_encode`, `gemv_i8` and
> `gemv_i8_r2`. Their bit-identity rows are OWED to the same run. Scratch is 0 B. VGPR is 23
> (was 17; hipcc 24), 19 (was 16; hipcc 18) and 24 (was 21; hipcc 23). Left for the integration
> perf pass, per the coordinator's brief: `gemv_f32`'s `for_i32`, and `hash_rows`' atomic. An
> agent-scope XOR needs either inline asm (`global_atomic_xor_b64`, which hides the operation
> from the compiler) or a `syncscope("agent")` route. `core::sync::atomic` and `core::intrinsics`
> both lower to the default system scope in nightly 1.102, and no other route was tried.

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
| `attn_res` (1, 9, 7168, 9) — Phase 0, owed *(paid 2026-11-19: 0.995 / 1.000 on the integration lease, Step 4)* | 53.62 | 57.43 | 1.071 | 1.013 (unplanted) |

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
  - **K3: OPEN.** Its single expert reads 1.070 in the only valid sample. *(Closed 2026-11-19:
    1.034 / 1.041 on the integration lease's A/A′, not slow — Phase 1 integration, Step 4.)*

### Phase 1 integration (2026-11-19, branch `wave/rk-integrate`)

**Step 1 — the four lanes merged** onto `d6a6f1b`, in the order L4 (fast-forward, `464237f`),
L3 (`600447d`), L1 (`3e027e2`), L2 (`3b6b936`). After each merge the build, clippy (both
arms), jscpd and the deviceless suite (`--no-fail-fast`) were run, and docs, hook_guard,
kernel_coverage, line_limit and matrix were each confirmed to have run green. The only failure
was CodeScene on the inherited list of files below 10/10: 31 files, then 30 once L2's merge took
`hip_attn.rs` off the list.
Logs: `/var/cache/rivoli/scratch/rk-integrate/gates/<step>/`.
- **Censuses after L2:** 84 Rust kernels, 66 rows (25 kernels rowed via `LAUNCHED_BY`), 1
  argued exemption; 66/66 launchers with oracles; 53 with production callers, 13 classified.
  `REF_SOURCES` names all 13 tagged `.hip` files. hipcc's `KERNELS` is **empty**; the hipcc/`ar`
  half of `build.rs` compiles nothing and stays until Phase 2.
- **What the merges had to reconcile, beyond list unions:**
  - Three lanes had each invented an integer-output `Out` variant: L3's `Exact(Box<Out>)`, L1's
    `PoisonBits`/`ZeroBits` and L2's `Words(Vec<u8>)`. They are one 2×2 now:
    `Poison`/`InPlace` (f32, NaN as a class) and `PoisonBits`/`InPlaceBits` (bit for bit).
    L2's per-output `moved_by_out` rule and per-shape tally apply to every row.
  - L1 and L2 had both written `inputs::bytes`, drawing different LCG bits (56..63 against
    40..47). Both are kept, as `bytes` and `weight_bytes`, so each lane's recorded CPU plant
    rates still describe the cells its rows launch.
  - Both L1 and L3 shipped a `plant_models.rs`; L1's is `linalg_plants.rs` now.
  - `fp8_lut` existed twice, as identical copies in `linalg.rs` and `mla.rs`; it is one function
    in `gpu.rs` now. `nasty()` and `rbf16` are deduplicated across lanes.
  - With no hipcc row left, `launchers!`'s `->` arm, its `extern "C"` emission and `abi_ty!`
    are deleted. `hip.rs`'s last extern block went with L1's `hash_rows`, and its jscpd region
    now ends after the memcpy/fill wrappers.
- **Plants checked by hand:** all 24 patches under `tests/bit-identity/plants/` (the 8 probe
  plants, `barrier` and 15 launcher-row plants) apply to the merged tree, two of them only after
  being regenerated with the same defect:
  - `linalg.patch` broke on the integration's own `fp8_lut` import;
  - **Phase 0's `situ` probe plant broke on L1's `pub fn tanh_ocml`.** No lane gate covers a
    probe plant's context, so it would have surfaced only as a refused build on the next lease.

  Phase 0's rope plant had existed only as a prebuilt binary
  (`scratch/rk-infra/plant/bit_identity-rope-unfused`). It is
  `tests/bit-identity/plants/launch/rope.patch` now, the `lo` write unfused, matching
  `plants.rs::rope_unfused_plant_is_adversarial_on_cpu`.
- **Declined at the merge reviews, each recorded for its owner:**
  - Phase 2 owns: the dead hipcc half of `build.rs` and the 9 headers under
    `crates/backend/kernels/` (the reference reads them from the tag, never the tree); the
    `launchers!` matcher's now-decorative `as`/`=>` annotations; and the census's `->` arrow.
  - L2 owns: the four `MoeRange` literals inside `hip_blocks.rs`'s exempt region, and the gdn
    `rep` computed in the row.
  - A follow-up owns: one shared plant-threshold helper in place of four.
  - L1's and L2's launcher-row plants sit at the plants root, not `plants/launch/`, so
    `--build-probes --plant=moe` would build a probe object no row reads. That is a usability
    hole, not a gate hole: no arm uses that route.

**Step 2a — `gpu::for_i32` is `nsw`** (coordinator decision, L4's prepared patch). The
increment is `i.unchecked_add(stride)`, UB only if `end + stride > i32::MAX + 1`. For a C loop
striding by a constant (`SUBW`, `*_BLOCK`) that is exactly clang's `nsw`. **For a C loop striding
by `blockDim.x` it is not** (review): `int + unsigned` is unsigned in C, so hipcc emitted no
`nsw` there (`kvcompress.hip`, `mla.hip`, parts of `linalg.hip`, `attn.hip`'s combine). The Rust
increment promises more than the C did. It is sound because every stride is a block size and
every `end` a launcher-guarded extent, but those kernels' loops no longer mirror hipcc's
codegen. A `debug_assert!` states the bound. It is compiled out, because the device crate always
builds `--release`. Static census of all 84 kernels, before
and after (`bin/census.py`, `isa/nsw-census.txt`):
- **The FP-op census is identical in 84/84 kernels** (`bin/fpcensus.py`: every
  fma/mul/add/sub/max/min/exp/rcp/sqrt/cvt/cndmask mnemonic, VOPD halves included). The change
  moves only address arithmetic, so the bits are argued unchanged; the lease re-scores them.
- **Total static instructions:** 40495 → 40464. 24 kernels moved; the largest, as static
  instruction deltas: `mla_latent_attend` −142 (1249 → 1107; VGPR 47 → 45),
  `rmsnorm_weightless_batch` −24, `rmsnorm_centered_single` −15, `rmsnorm_batch` −10,
  `gemv_f32`/`_r2` −8 each; `kv_compress_decode` +38, `kv_compress_prefill` +37, `index_topk`
  +33, `hc_pre` +31, `rope_adjacent` +11, `act_quant_f4_rotated` +10, `gemv_i8` +10. The two
  kv_compress kernels are blockDim-strided in the C (above). Every growth is loop restructuring
  with the FP census unchanged; timing judges them.
- **Occupancy: no kernel changes waves/SIMD.** One crosses the 24-VGPR boundary of the
  occupancy table (the census's 24-register rounding, `bin/census.py`): `gather_attn_shared_kv`
  23 → 25 VGPR, as L4 found. Below 96 VGPRs every kernel sits at the
  16-wave cap, so the granule costs nothing here. The kernels nearest the 96 edge did not move:
  `hc_pre` stays at 94, and the MoE int4 R=2 pair sits above it, unchanged.

**Step 2b — the `LAUNCHED_BY` rule loosened** (coordinator decision). A launcher's top-level
body must still name each listed kernel's static as a whole identifier. It no longer has to
spell `run(&STATIC,` or `STATIC.launch`: a `run(`, `run_2d(` or `.launch` anywhere in the body
suffices. Trailing `//` comments are now dropped along with comment lines, or a
`// STATIC` after code would count as naming it.
- **Red-proofed:** five text plants, each red on exactly the entries it targets, and the clean
  tree green (`gate-red-proofs.md` §19, integration rows). A sixth records the class the
  loosening gives up, by decision: a body that names a static in code it never launches while it
  launches another (exit 0). The bit-identity row is what would catch that.
- **L2's int4 launcher is back to the natural form:** a `match` on `(down, r2)` picks the
  static into a local, which is launched once. jscpd measures 0 clones: the VQ twin launches
  in each arm, so the token runs differ. Bits are unaffected, since the same static is
  launched with the same arguments.

**Step 3 — the performance pass (deviceless; every change is re-scored and re-timed on the
integration lease).** The method was to read each owed row's loops in the shipped code object
against hipcc's `--cuda-device-only -S` of the tag, with backward branches found and body
mnemonics counted (`scratch/rk-integrate/bin/{kisa.sh,loops.py}`). The bits argument for every
change is the FP-op census (`bin/fpcensus.py`): identical where a loop kept its shape, and where
LLVM now unrolls, different only by the extra unrolled copies, with each element's or entry's
operations unchanged in kind and order (no fp reassociation without fast-math).

Three causes were found, lettered on from L1's (a) and (b), and all three generalize beyond the
rows that exposed them:
- **(c) Rust's checked integer division.** `/` and `%` on a runtime divisor carry a divide-by-zero
  (and, for signed types, an overflow) panic path: an `s_trap` block, plus the per-lane
  exec-mask branching the check needs, inside whatever loop holds it. C division is UB on
  exactly those inputs, so hipcc emits neither.
  - `core::intrinsics::unchecked_div`/`unchecked_rem` now carry the C's contract, as
    `linalg`/`fwd`/`headtail`/`residual` already did. Each site cites the launcher guard that
    makes its divisor nonzero. (A generic `cdiv<T>` wrapper was tried and dropped: calling the
    intrinsic through a type parameter ICEs this nightly, and the wrapper only added a second
    convention.)
  - Applied to `attn/gqa.rs`, `attn/mla.rs`, `mla.rs` and `kvcompress.rs`.
  - Traps were removed in `gqa_attend` (7), `gqa_block_attend` (6), MLA attend (2), the fp8
    MLA family (3 each),
    `gemv_fp8_bf16` (7), and the compressor (deposit 4, decode 2).
  - Still carrying one or two traps each: the MoE range kernels, `gdn_recurrent_f32`, the block
    scorers (`indexer.rs`'s `id / n_comp`) and `moe.rs`'s `r / len`. Those rows are at parity
    and were left.
- **(d) A short-circuit `||` kept as divergent branches.** `f2e2m1`'s
  `a > m || (a == m && i odd)` compiled to `v_cmpx_eq_f32` + `s_cbranch_execz` at the three odd
  midpoints, where clang selects. Now `(a > m) | ((a == m) & odd)`, the same truth table. This
  is a prelude codec: the probe arm's exhaustive `ex_f2e2m1` sweep re-scores it.
- **(e) An iterator adapter.** `(0..ents).filter(live)` became a nested find-next scan on 64-bit
  counters, where hipcc unrolls `for (int j…) { if (!live) continue; … }` 8×. Now
  `kvcompress::each_live`, an `int` loop in the C's shape.

Static instructions, hipcc vs before nsw (`pre`) vs now (`post`). A larger `post` can be the
unrolling hipcc does too (`kv_compress_decode`); the last column says which:

| row (lease ratio) | hipcc | pre | post | what changed, and whether the loop shape now matches hipcc |
|---|---|---|---|---|
| `gqa_attend` (1.14) | 689 | 915 | 684 | (c): the tile load's `idx / d`, `j % ring_cap`; exec-mask structure = hipcc's (53 `s_and_saveexec` each) |
| `gqa_block_attend` (1.07) | 653 | 802 | 644 | (c), as above |
| `mla_attend` (1.20) | 979 | 1249 | 994 | nsw (dot loops = hipcc's), then (c) in the tile load (`/ kvl`, `/ rope`) |
| `gather_attn_shared_kv` (1.21) | 720 | 918 | 897 | values loop on the C's `int` counter (L4's lead; drops `v_cmp_lt_u64`). **No cause found:** per iteration, the logits, exp and values loops match hipcc's (Rust unrolls the j-loop 2× with the same per-j work); 22 `s_nop` hazard pads, against hipcc's 0, sit outside the hot loops |
| `gemv_f32` (1.09–1.10) | 101 | 111 | 103 | nsw: lane loop = hipcc's (one extra `s_delay_alu`) |
| `gemv_i8` (1.06–1.08) | 170 | 178 | 188 | nsw + L1's dword-loop change: dword loop 32 vs 30 (a 64-bit scalar offset bump), tail loop identical |
| `gemv_fp8` split-K (≤ 1.09) | 278 | 324 | 324 | main loop 52 vs 51, tail identical; L1's unpriced combine difference (`ds_load_2addr_b32` against `ds_load_b128`) left |
| `act_quant_f4_rotated` (1.47) | 377 | 394 | 415 | (d): the round trip is branchless and LLVM now unrolls it 2× unaided, 103 vs hipcc's 102 instructions per two elements. Remaining: the amax loop reads LDS as `ds_load_2addr_b32` where hipcc uses `ds_load_b128` |
| `kv_compress_decode` (1.22) | 1017 | 841 | 1144 | (e) + (c): entry loops unrolled 8× like hipcc's (max 95/93, sum 101/99, weighted sum 117/113) |
| `kv_compress_deposit` (1.06) | 482 | 573 | 499 | (c) |
| `mla_value_fp8` (1.055) | 424 | 516 | 445 | (c) |
| `attn_res` (1.04–1.07) | 725 | 1590 | 1583 | **left:** Phase 0's 16-slot `each_src` is what keeps `p[16]` out of scratch (144 B → 0, 1.38× → 1.06×); within noise for its size |
| `hash_rows` (1.8–1.9, `--divergence-log` only) | 169 | 262 | 262 | **left:** system-scope `fetch_xor` = a CAS loop; an agent-scope XOR needs inline asm or a syncscope route, neither available in nightly 1.102's `core`. Diagnostic-only |
| `moe_down_i4_r2` (int4 nrow 2, 1.070) | 802 | 801 | 801 | **left:** VGPR 100 against hipcc's 95, 12 against 16 waves. A prelude property (`dot_nib_wave_r::<2>`); any change re-scores every nib caller |
| K3 fp4 single expert (1.070, one sample) | — | — | — | **left:** no ISA difference was identified in L2's census; the lease's A/A′ spread decides whether it is real *(decided: not slow, Step 4)* |

Four kernels outside the table also changed:
- `gemv_fp8_bf16` 906 → 823 and `mla_absorb_fp8` 1044 → 1002, both from (c). Neither was owed:
  `mla_absorb_fp8` already ran faster than hipcc and `gemv_fp8_bf16` at parity (1.017).
- `rope_adjacent` 192 → 156 and `act_quant_f8_prefix` 422 → 367, also (c).

None of these is a timing claim. Each row's ratio is judged on the lease, against the A vs A′
spread.

**Step 4 — the integration lease (2026-11-19, commit `f926d2e`, rh-anine).** `lease.sh` ran the
whole lease from one `setsid`-detached driver, with its own PGID and SID, from 16:21 to 20:48.
- **Waiting and builds:** waits were on progress content lines, never on process absence.
  Every binary was built outside the lock before the GO: the dev build checked at 0 units to
  compile, `bit_identity` Fresh, the 13 plant binaries (`git archive f926d2e` + patch, in their
  own target dir, with restoration checked by `diff -r`), the probe objects with current
  stamps, and the release `rivoli`.
- **Witness:** **26 arms, none discarded; every witness 0 B; GTT 17.8 MiB before every arm; 0
  units compiled inside the lock.** GTT read 17 MiB (the witness rounds down) before every arm;
  18702336 B exact around D. Host loadavg stayed at 1.1–3.6, except at the end of R-indexer and
  into R-blockindex (27.8; a foreign `java` process, seen in `ps` during the lease, CPU only and
  not a GPU tenant) and at the end of D (11.5).
- **Evidence:** `/var/cache/rivoli/scratch/rk-integrate/lease/` (`progress`, `<arm>.out`,
  `timing-AA2.txt`, `D.gtt`, `D-tmp/`); per-arm logs are in the `…/bit-identity/<stamp>` each
  `.out` names.

| arm | verdict | rc | wall |
|---|---|---|---|
| A | GREEN: 66/66 rows BIT-IDENTICAL | 0 | 15.3 min |
| A′ (A again, the timing control) | GREEN: 66/66 | 0 | 15.4 min |
| 13 launcher-row plant arms (the 15 patches, L2's three in one binary) | each GREEN (expect-red), exactly its own rows red | 0 each | 15.0–16.9 min each |
| `--probes` | GREEN: every probe row identical | 0 | 0.9 min |
| 8 probe plants | each GREEN: red only where the defect reaches | 0 each | ≤ 0.9 min each |
| W (full battery, `--test-threads=1`) | 720 passed, 2 failed: CodeScene (the 30 base files) and `kernel_qwen_indexer::the_block_key_pipeline_matches_the_anchor` — a TEST defect, below | 101 | 18.9 min |
| D (`determinism-glm.sh`, `--arena-refresh`, ngen 1024) | GREEN: ids identical to 1024 | 0 | 13.5 min |

- **Bits.** The combined tree is byte-identical to the tagged hipcc on every launcher row. That
  includes every Step 2/3 change: nsw, the unchecked divisions, the `f2e2m1` boolean (also
  through the probes' exhaustive `ex_f2e2m1` sweep), `each_live` and gather's `int` counter.
- **Plants.** Every plant re-proved itself individually on the combined tree, with cell counts
  equal to the lanes' own leases and CPU models:
  - L1: `gemv_f32` 196, `gemv_i8` 172, `sigmoid_gate` 256, `hc_head_collapse` 167;
  - L3: `index_topk` 126, `index_score_blocks` 203, `gemm_bf16` 266, `mla_absorb_fp8` 256;
  - L2 (one binary, all three plants): exactly its five rows;
  - L4: `mla_attend` 551, `gather_attn_shared_kv` 187, grid-y's `mla_attend` 384, and the GQA and
    MHA rows red as planted;
  - rope 275, on its first run as a patch;
  - the probe plants' red-group counts, 6/18/4/30/1/34/28/15, are Phase 0's exactly, so the
    regenerated `situ` plant works.
- **D.** GTT read 18702336 B both before and after (rise 0); witness a/b 0 B; `slot_stalls 0`;
  decode at 2.86 and 2.76 tok/s (Phase 0's D ran at 2.72 and 2.96). Like every D, this bounds
  the nondeterminism rate; it does not prove determinism.

**The A vs A′ noise floor** (`timing-AA2.txt`; the minimum of three 200-launch batches per
shape):
- Most rows of 10 µs and up repeat within ±0.005; the widest are `kv_compress_prefill` 0.037,
  `moe_expert_range_i4` nrow 1 0.028, `argmax` 0.026, `vq_encode` 0.018 and `index_topk` 0.017.
- The ~2 µs rows swing by up to 0.20: `swiglu` 0.128, `swiglu_clamped_bf16` 0.181,
  `situ_glu_f32` 0.145, `embed_bf16_row_bcast` 0.197, `rope_adjacent` 0.120. That is the
  "unexplained swing" of every earlier lease, now measured as a per-row floor.
- **Rule (coordinator, fixed before the result):** a row is slow only if its mean A/A′ ratio
  exceeds 1 + max(0.05, its A/A′ spread).

| owed row | lease ratio before | A / A′ now | verdict |
|---|---|---|---|
| `mla_attend` | 1.20 | 1.003 / 1.004 | not slow |
| `gqa_attend` | 1.14 | 0.988 / 0.988 | not slow |
| `gqa_block_attend` | 1.07 | 1.009 / 1.008 | not slow |
| `act_quant_f4_rotated` | 1.47 | 0.959 / 0.957 | not slow |
| `gemv_f32` | 1.09–1.10 | 1.036 / 1.034 | not slow |
| `gemv_i8` | 1.06–1.08 | 1.021 / 1.020 | not slow |
| `gemv_fp8` split-K | ≤ 1.09 | 1.018 / 1.012 | not slow |
| `vq_encode` (L1's `i32` fix) | 1.61 | 0.967 / 0.949 | not slow |
| `kv_compress_deposit` | 1.06 | 1.037 / 1.020 | not slow |
| int4 range nrow 2 (`moe_down_i4_r2`) | 1.070 | 1.035 / 1.036 | not slow |
| K3 fp4, single expert | 1.070 | 1.034 / 1.041 | not slow |
| `attn_res` | 1.04–1.07 | 0.995 / 1.000 | not slow |
| the gated combines | 0.96–1.17 | 0.98–1.16 | not slow (inside a 0.13–0.18 spread) |
| **`gather_attn_shared_kv`** | 1.21 | 1.109 / 1.106 | **slow**: no loop-shape cause found; 22 `s_nop` hazard pads outside the hot loops unpriced (Step 3) |
| **`kv_compress_decode`** | 1.22 | 1.060 / 1.070 | **slow**: loops now hipcc-shaped, the rest unexplained |
| **`mla_value_fp8`** | 1.055 | 1.060 / 1.062 | **slow**: float census identical to hipcc's (L3) |
| **`hash_rows`** | 1.8–1.9 | 1.932 / 1.897 | **slow**: the system-scope atomic, `--divergence-log` only |

Also faster than hipcc, as before: `mha_attend` 0.27×, `gdn_recurrent_f32` 0.42×, the block
scorers 0.18×.

**W's second failure: a race in the test, confirmed and fixed (coordinator-directed).**
- **The race.** `kernel_qwen_indexer.rs` launched `index_pool_norm_f32` on `s` (from
  `common::stream()`, non-blocking). It then ran `launch_rope_split_half`, which takes no
  stream and so runs on the null stream, in place on the same `kbar` rows with no join. The null
  stream does not order against a non-blocking stream.
- **Why it surfaced now.** The kernel was byte-identical to hipcc in A and A′. The hipcc build
  passed by winning the race; the Rust launch path lost it (`pooled_keys` 7.1479464e-1 against
  1.14e-4).
- **The fix.** A `device_sync()` after the pool-norm. The same join also went after the query
  RoPE, the reverse order on `qrows`, which had been ordered only by two blocking uploads that
  happen to sit between the launches. The tolerance is untouched.
- **Confirmed on the device** (`scratch/rk-integrate/confirm/`, six witnessed arms under the
  flock, all witnesses 0 B):
  - the unfixed binary (W's own, copied before the fix) failed 2 of 3 runs, at the same
    7.1479464e-1;
  - the fixed binary passed 3 of 3. That is thin on its own (at the unfixed binary's 1-in-3 pass
    rate, three straight passes would happen about 4% of the time), and the fix stands first on
    the code argument.
- **The sweep for the class** (deviceless, over every test and `crates/engine/src`,
  `crates/cli/src`) found no other race. Every engine cross-stream edge is ordered, all by an
  explicit `device_sync` except the one below, which a blocking readback orders.
- **OWED hardening, not a defect today:** `glm/mlp.rs`'s `launch_moe`. The compute and miss
  streams read the null stream's writes of `xn` (and the gate GEMV) with no explicit join; only
  the blocking gate-logits D2H (`copy_out_into`) orders them. V4 (`v4/moe.rs`) and K3
  (`k3/forward.rs`) make that an explicit join. Remove the readback and GLM would race.

**Left owed:**
- the four slow rows above;
- the `glm/mlp.rs` join; *(done 2026-11-20, Phase 2)*
- Phase 2: the dead hipcc half of `build.rs`, the 9 headers, the `launchers!` matcher's
  decorative annotations and the census's `->` arrow; *(2026-11-20: all but the annotations
  done in Phase 2, which records why those stay)*
- L2's `MoeRange` literals;
- one shared plant-threshold helper;
- from Step 1: L2's gdn `rep` computed in the row; the L1/L2 launcher-row plants at the plants
  root rather than `plants/launch/`;
  > **Closed 2026-11-23 (`wave/open-items`):**
  > - **`MoeRange` literals — declined, with the reason.** The four literals spell every field by
  >   name inside `hip_blocks.rs`'s `launchers!` rows (a jscpd-exempt invocation region by
  >   design). The only shorter form is a positional `MoeSpan::of(hidden, inter, e_start,
  >   e_count)`: four same-typed integers whose transposition would compile — the class the Phase 0
  >   merge review refused (`Geometry = (u32, u32, u32)`). Named fields are the safer spelling.
  > - **One shared plant-threshold helper:** `crates/engine/tests/bit_identity/plant_bar.rs`
  >   (`Rate`, `Bar`, `MOST` = 9/10, `demand`) replaces the four copies in `plants.rs`,
  >   `linalg_plants.rs`, `attn_plants.rs` and `plant_models.rs`; models that argue a lower bar pass
  >   their own `Bar`. Red proof: `MOST` raised to 11/10 reddened exactly the 11 models that use it,
  >   the four with their own bar or assert held.
  > - **The gdn `rep`:** derived in `launch::recurrent::rivoli_gdn_recurrent_f32` after
  >   `gdn_checked` has refused `qk_heads <= 0` (1001) and a ragged ratio (1003); the row passes 0.
  >   It had been computed in the row, before any guard, as `v_heads / qk_heads.max(1)` — the
  >   `.max(1)` hid a zero, and the kernarg doc claimed the launcher derived it after the guard.
  >   Same value whenever the guard passes; device check is the bit-identity `gdn_recurrent_f32` row.
  > - **Plants layout:** the 15 launcher-row plants (L1, L2, L3, L4, rope) all live in
  >   `tests/bit-identity/plants/launch/` with repo-root paths; the root holds only the 9 probe
  >   plants. `tests/bit-identity/build-launch-plants.sh` (in-tree port of the integration's
  >   scratch script) builds the planted binaries from `git archive HEAD`, and
  >   `crates/cli/tests/plants.rs` holds both tables equal to the patch files at both ends and
  >   dry-runs every patch (red-proofed: an unlisted patch, a probe plant with no `plant_spec` row,
  >   and a drifted context each reddened exactly their own test). The rope plant built through the
  >   script (`e791d0c0…`).
- CodeScene: the 30 inherited files below 10/10, unchanged by this branch.

### Perf follow-up (2026-11-19, branch `wave/rk-perf`, base `51ab275`)

The four rows Step 4 left slow. Method as Step 3: the shipped code object (built from
`crates/kernels` with `device.rs`'s flags, `scratch/rk-perf/bin/kb.sh`) read loop by loop against
hipcc's `-S` of the tag (`scratch/rk-integrate/ref/`), plus a new per-loop VALU/SALU/VMEM split
(`bin/loopclass.py`, `bin/cls.py`). Every experiment's ELF and per-kernel ISA are in
`scratch/rk-perf/{elf,isa}/` (`base` = `51ab275`). Each row had its own cause:

| row (A / A′) | cause | change | static instructions, hipcc / before / after |
|---|---|---|---|
| `gather_attn_shared_kv` (1.109 / 1.106) | **The barrier's asm hid uniform loads.** The values loop reads `ix[j]`, uniform per block. hipcc loads it with `s_load_b32` (scalar cache, `lgkmcnt`), Rust with `global_load_b32` + `v_readfirstlane` (a vector-memory round trip per slot, 640 slots at the timed shape, then the dependent `kv` load). AMDGPU makes a uniform global load scalar only if `isClobberedInFunction` finds no clobber above it; that analysis skips fences and barrier intrinsics, but `sync_threads`'s asm memory clobbers counted as writes to every address. | `sync_threads`: the asm is `nomem`, and two `compiler_fence(SeqCst)` carry the ordering (argued in `prelude/reduce.rs`). The values loop is now hipcc's per slot (`s_load_b32`, `s_cmp`/`scc` branch), 2× unrolled. | 720 / 897 / 890 |
| `kv_compress_decode` (1.060 / 1.070) | **A lost `__restrict__` serialised the window slide.** Pooling, finish and every other loop matched hipcc or were leaner (the tree is unrolled where hipcc loops on `blockDim`). The slide was not: hipcc issues `kv_state[n+i]` and `score_state[n+i]` before either store, while Rust waited `vmcnt(0)` between them, because the score load may alias the kv store through raw pointers. That is one more serial global round trip per element, 16 per thread, in a one-block kernel. L2's `gated_delta` cause again (loads before stores). | Both loads first, then both stores. The buffers are distinct and `[n, 2n)` never meets `[0, n)`, so the values are unchanged. The loop is now hipcc's: two loads, `vmcnt(1)`, store, `vmcnt(0)`, store. | 1017 / 1144 / 1158 |
| `mla_value_fp8` (1.060 / 1.062) | **`usize` where the C has `int`, in a VALU-heavy kernel.** Estimated from static counts on the timed path: 16384 waves of about 250 VALU each keep the 80 SIMDs issuing for roughly two thirds of the 27 µs, so VALU count tracks time. Rust ran about 22 more VALU per wave than hipcc (≈ 9%): 10 in the prologue (`head`/`j` from the 64-bit `r / vh` stayed 64-bit, so `head · (nope + vh)` became full 64×64 products) and 3 per iteration of the 4-iteration dword loop. The C's `int head` lets clang use 32×32→64 `v_mad_i64_i32`. | `head`, `j` (and `nope + vh`) as `i32`, as the C has them. Same values: `head < h`, `j < vh`, all `i32` kernarg fields. Prologue VALU 125 → 115, hipcc's count; multiply count 23 → 18, hipcc's. | 424 / 445 / 448 |
| `hash_rows` (1.932 / 1.897) | **Atomic scope**, as Step 3 found: core's system-scope 64-bit `fetch_xor` expands to a load + `global_atomic_cmpswap_b64` retry loop (32-bit halves expand the same way, measured). | `fwd.rs::agent_xor`: one `global_atomic_xor_b64` in asm, hipcc's instruction, no return. The memory-model argument is at `fwd.rs::agent_xor`. | 169 / 262 / 242 |

- **What remains, by row (ISA, not timing):**
  - `mla_value_fp8`'s dword loop is still 31 VALU against hipcc's 28. The three extra are a
    `v_add_nc_u32 v, 0, v` on three of the four LUT reads: the dynamic-LDS base is a link-time symbol (resolved to
    0), so codegen cannot fold it the way hipcc folds a static `__shared__` array's offset. That
    is structural for every kernel that addresses dynamic LDS through computed indices;
    Rust has no LDS statics. The 64-bit loop counters and hipcc's sign extensions roughly cancel:
    an `i32` loop in `fp8_dot_strided_r` (`expC`) came to 30, and was left out, because it changes
    the shared fp8 prelude under every `gemv_fp8*` row for about 4 VALU per wave here.
  - `kv_compress_decode`, `gather_attn_shared_kv` and `hash_rows`: no further difference found
    (`hash_rows`' sample loop is hipcc's, 43 instructions each). Static counts stay above hipcc's
    because Rust unrolls the block-size-constant LDS trees that hipcc loops on `blockDim`.
- **The barrier change reaches every kernel.** Totals: FP-op census identical in 79/84 kernels.
  The other five, all argued:
  - `gated_delta_recurrent_f32` moves 2 `v_fma` and 1 `v_add` to `s_fmac_f32`/`s_add_f32`: the
    same IEEE ops on SALU, now that their uniform operands load scalar. hipcc does this itself,
    and the base tree's bit-identical `hc_pre` already had 32 `s_fmac_f32`.
  - `hc_pre` only swaps `v_fma` for `v_fmac` encodings, 350 each.
  - `gdn_recurrent_f32` and the two `mla_value_fp8` rows differ in `v_cndmask`: integer selects,
    in the division and in loop control. They are not FP values.
- **Occupancy:** no kernel changes waves. `mla_value_fp8` goes from 24 to 25 VGPR, still 16 waves.
- **Bits: OWED to the lease.** The argument: every change moves address arithmetic, load order,
  or a load from vector to scalar memory; no float operation changes value or order, and the
  three that move VALU → SALU are the same IEEE ops (above).
  > **SETTLED 2026-11-19 (the lease, below):** 66/66 rows BIT-IDENTICAL in A and A′.
- **Plants.** All 24 patches still apply under `patch -p1 -F0`.

**The lease (2026-11-19, commit `fe02120`, rh-anine).** One `setsid`-detached driver,
`scratch/rk-perf/lease/lease.sh`, ran 22:08:59–23:28:34 +01:00. Every binary was built outside
the lock, and 0 units compiled inside it (`W.out`). No arm was discarded: every arm that reached
the device had an empty witness, and GTT read 17 MiB before each (17.8 MiB exact around D).

| arm | verdict | rc | wall |
|---|---|---|---|
| A | GREEN: every row BIT-IDENTICAL | 0 | 15.1 min |
| A′ (timing control) | GREEN: every row BIT-IDENTICAL | 0 | 15.4 min |
| launcher plants, one binary (fwd, launch/kvcompress, launch/mla, attn-dense) | GREEN (expect-red): exactly `gather_attn_shared_kv` 187, `mha_attend` 209, `gemm_bf16` 266, `mla_absorb_fp8` 256, `sigmoid_gate` 256 cells red | 0 | 15.4 min |
| `--probes` | GREEN: 222/222 probe rows identical | 0 | 0.9 min |
| `--plant=reduce` | GREEN: `k_wave_sum` red 64/64, the must-hold rows held | 0 | 2 s |
| `--plant=barrier` | **refused at setup** (`no plant_spec row for 'barrier'`): that plant's arm is owed (its judge, a per-kernel `s_barrier` census, is not in the tree). Listing it was a planning error in the READY post. Not rerun. *(Its judge landed 2026-11-20: Phase 2, below.)* | 2 | 0 s |
| W (full battery, `--test-threads=1`) | 721 passed, 1 failed: CodeScene, the same 30 inherited files as Step 4's W (lists diffed equal) | 101 | 19.2 min |
| D (`determinism-glm.sh`, `--arena-refresh`, ngen 1024) | GREEN: ids identical to 1024; GTT rise 0; `slot_stalls 0`; 2.73 and 2.82 tok/s | 0 | 13.5 min |

- **In place of the barrier arm, deviceless:** the census that plant's judge would read. The
  per-kernel `s_barrier` and `buffer_gl0_inv` counts are identical before and after in all 84
  kernels, 39 of which have barriers (`lease/barrier-census.txt`). W and D are the race arms the
  coordinator added for the barrier change: D is green, and W's one red is CodeScene.
- **The four rows, by the fixed rule** (closed when the mean ratio is ≤ 1 + max(0.05, A/A′
  spread)):

  | row | before (Step 4 mean) | A / A′ | mean | verdict |
  |---|---|---|---|---|
  | `gather_attn_shared_kv` | 1.107 | 1.000 / 0.994 | 0.997 | **closed** |
  | `kv_compress_decode` | 1.065 | 0.963 / 0.973 | 0.968 | **closed** |
  | `mla_value_fp8` | 1.061 | 1.021 / 1.018 | 1.019 | **closed** |
  | `hash_rows` | 1.914 | 0.908 / 0.910 | 0.909 | **closed** (now faster than hipcc) |

- **Every other row against Step 4's table** (`scratch/rk-perf/lease/cmp.txt`, `bin/cmp.py`).
  A row "moved" when its mean changed by more than the larger of the two leases' A/A′ spreads.
  Of the 71 timed shapes, none is slow by the fixed rule. Three ~2 µs rows sit above 1.05 and
  pass on their spread alone (A′ spiked about +0.17 on each): `swiglu` 1.085, `index_append`
  1.077, `index_pool_push` 1.084.
  - **Faster:** 16 moved faster, the four above and 12 more, largest `logit_softcap` −0.022.
  - **Slower, three rows, all still at parity** (the move threshold is the larger spread, 0.001
    for the fp4 row):
    - `hc_post` 0.923 → 0.961 (+0.037; a 2.4 µs row whose spread is 0.024–0.025);
    - `act_quant_f4_rotated` 0.958 → 0.968 (+0.010);
    - `moe_expert_range_f4` single expert 1.018 → 1.020 (+0.002).

### Phase 2 (2026-11-20, branch `wave/rk-phase2`, base `5dd417b`)

Deviceless half; the lease is recorded below it. Logs: `/var/cache/rivoli/scratch/rk-phase2/`
(`gates/`, `census/`, `deviceless/`).

**1. hipcc out of the build.** `crates/backend/build.rs` keeps the libamdhip64 link and the
device-crate build (`build/device.rs`, which now reads `RIVOLI_OFFLOAD_ARCH` itself); the
`KERNELS` list, `headers()`, `Toolchain` and the `librivolikernels.a` archive are gone, and so is
`crates/backend/kernels/` (the 9 headers). The bit-identity reference never read the tree: the
launcher arm, the probe arm and the new census all `git archive` the tag (`ref_tree`), and the
probe build reads only `crates/kernels`, `crates/abi` and the `device-rustflags` file
`build/device.rs` writes.
- **The embedded code object did not move:** sha256 `318f0bfb…` in this branch's dev and
  release builds, in a fresh target dir, and in the `rk-perf` target dir whose object the
  `fe02120` lease scored 66/66 bit-identical.
- **The rocm build needs no hipcc, shown:** a `hipcc` that exits 97 first on `PATH`, a freshly
  deleted target dir, `cargo build -p rivoli-backend`: rc 0, the device crate built.
- **The deviceless arm needs neither nightly nor ROCm, re-shown** (Phase 0a's recipe): a
  `RUSTUP_HOME` holding only the stable toolchain, the failing `hipcc` first on `PATH`,
  `RIVOLI_DEVICE_LIBS=/nonexistent`, a freshly deleted target dir, `cargo test --workspace
  --no-default-features --no-fail-fast`: 516 passed, 1 failed (CodeScene, below). Red proof that
  the environment lacks the device toolchain: the same environment building `-p rivoli-backend`
  (rocm) fails in `build/device.rs` with `toolchain 'nightly-2026-11-18…' is not installed`.
- **Every census still examines its population:** kernel_coverage 66/66 launchers with oracles,
  53 called + 13 classified, 35 prelude probes; the scanner lost its `->` arrow (no row has
  carried one since the integration).
- **Docs and comments:** about 50 files whose text said the C was in the tree, was a build
  input, or located live behaviour in a deleted `.hip`, now point at the Rust (red-proof recipes
  that moved carry a dated note that their recorded red was on the hipcc kernel). A bare
  `kernels/<file>.hip` citation names the tag, by a CLAUDE.md convention line. The jscpd exempt
  count stays 10 (no marker moved; `docs.rs` derives it).
- **Declined:** the `launchers!` rows' `as`/`=>` type annotations, decorative since the `->`
  arm left. Removing them is 166 one-line edits across the five jscpd-exempt invocation files,
  where a parameter annotation and a kernarg struct field (`s: d.s as i32,`) look alike to any
  mechanical rewrite; it changes no behaviour, so it is left for an owner-approved pass.

**2. The barrier census** (`tests/bit-identity.sh --census`, the `barrier` plant's judge). It is
a script mode, not a cargo test, because its reference needs hipcc and its object the pinned
nightly, neither of which the deviceless cargo arm may assume. Per kernel, the static
`s_barrier` and `buffer_gl0_inv` counts of the probe object, held equal to the embedded object's,
against hipcc's objects of the tag. 12 kernels differ by construction and are pinned exactly
(`DIVERGE`: hipcc loops an LDS tree on blockDim where Rust unrolls the constant's). Clean: 46/46
rows SAME. `--plant=barrier`: exactly the 11 `block_sum_lds` callers red, 14 must-hold rows held,
scripted `GREEN (plant barrier)`. Eight further plants and stubs red or refused as designed
(`gate-red-proofs.md` §21).

**3. GLM's MoE join.** `glm/mlp.rs::route_layer` calls `device_sync()` right after the
gate-logits D2H: JOIN 1, as V4 and K3 have it. The null-stream writes (`xn`, the gate GEMV)
were ordered before the non-blocking MoE streams only by the readback. Placed before `submit`,
every stream is already idle, so it waits on no device work; its cost is priced by the decode
A/B below.
> **NOT PRICED 2026-11-20 (the lease, below):** the A/B bounds the whole migration, not the
> sync; a B vs B-without-the-sync arm would.
> **SUPERSEDED 2026-11-20 (Phase 2 follow-up):** that arm ran (~1.5 ms/token post hoc, the
> pre-registered rule not met), and the sync is now the `NullStreamDrained` token.

**Gates (deviceless):** fmt, the dev build (jscpd ran, 0 clones), clippy both arms rc 0;
`cargo test --workspace --no-default-features --no-fail-fast` 516 passed, 1 failed: CodeScene,
the same 30 inherited files as the `rk-perf` lease's W (lists diffed equal).

**The lease (2026-11-20, commit `b5ca13a`, rh-anine).** One `setsid`-detached driver,
`scratch/rk-phase2/lease/lease.sh`, 01:35:34–02:44:13 +01:00. Every binary was built outside
the lock (W's pre-check: 118 units Fresh, 0 compiled inside). No arm was discarded: every
witness file is 0 B and GTT read 17.8 MiB before every arm.

| arm | verdict | rc | wall |
|---|---|---|---|
| A (bit-identity, launcher rows) | GREEN: 66/66 rows BIT-IDENTICAL | 0 | 15.9 min |
| W (full battery, `--test-threads=1`) | 721 passed, 1 failed: CodeScene, the same 30 inherited files (lists diffed equal) | 101 | 19.5 min |
| D (`determinism-glm.sh`, `--arena-refresh`, 1024) | GREEN: ids identical to 1024; `slot_stalls 0`; 2.95 and 3.02 tok/s | 0 | 13.0 min |
| AB (decode A/B, six arms) | B not slower (below); each of the six arms rc 0 (the driver's own rc cannot fail) | — | 19.6 min |
| P (`tests/parity-glm.sh`, ngen 32, mem 100) | GREEN: 32 generated ids identical to the pinned reference, at its full length | 0 | 0.7 min |

**The decode A/B.** A = the pre-migration engine, tag `archive/hip-kernels` (`abd065f`) built
`--release` with hipcc in its own target dir (`rk-phase2-premig`); B = this branch, `--release`.
Same command for every arm: `/var/db/rivoli/glm52-vq3-full --bench 512 --mode int3-vq --attn
dense --max-mem 115` with determinism-glm's 317-byte prompt (71 tokens), stock flags. Interleaved
A1 B1 A2 B2 A3 B3, each flock'd and witnessed; nothing built between arms
(`scratch/rk-phase2/lease/ab/`).

| arm | decode s / 512 | tok/s | attend ms/tok | ffn ms/tok | fetch-wait ms/tok | wall ms/tok | host load before → after |
|---|---|---|---|---|---|---|---|
| A1 | 166.1 | 3.082 | 93.816 | 115.121 | 109.878 | 324.453 | 9.23 → 4.37 |
| B1 | 165.7 | 3.090 | 95.425 | 112.943 | 109.824 | 323.577 | 4.37 → 4.52 |
| A2 | 165.5 | 3.094 | 93.875 | 113.549 | 110.207 | 323.275 | 4.52 → 5.12 |
| B2 | 166.9 | 3.068 | 95.523 | 112.810 | 112.202 | 325.966 | 5.12 → 6.22 |
| A3 | 165.0 | 3.103 | 93.996 | 113.496 | 109.053 | 322.201 | 6.22 → 5.90 |
| B3 | 174.4 | 2.936 | 97.176 | 115.785 | 121.907 | 340.550 | 5.90 → 11.11 |

- **Every arm decoded the same 512 ids** (one md5 across all six) with the same
  236570 hits / 70030 misses: the pre-migration and migrated engines are token-identical here,
  and so is their fetch traffic.
- **P5, bytes/token (derived, the bench does not print it):** 70030 / 512 = 136.8 misses/token ×
  15335424 B/slot (the routed pool's slot size, logged at startup) = **2.10 GB/token of
  slot-bytes**, an upper bound on bytes streamed, identical in A and B; the migration did not
  move the currency.
- **Verdict, by the coordinator's pre-registered rule** (B is not slower if mean(B) ≥ mean(A) −
  max(spread A, spread B)): mean A 3.093 (spread 0.021), mean B 3.031 (spread 0.154); threshold
  3.093 − 0.154 = 2.939 ≤ 3.031, so **B is not slower**. The gap (−0.062 tok/s) is carried by B3,
  whose host load climbed to 11.1 by its end; against the B1/B2 mean it moved every bucket
  (fetch-wait +10.9, attend +1.7, ffn +2.9 ms/token), as host contention would, though there is
  no host-process witness. Post hoc and outside the verdict: B1 and B2 alone average 3.079.
- **One difference the floor does not hide:** B's ATTEND bucket is above A's in every pair,
  ~1.6 ms/token (+1.7%) in the two unloaded ones, where FFN is 0.7–2.2 ms lower and the wall +0.45%, inside
  A's own 0.7% spread. **Unattributed**, three candidates: GLM's new JOIN 1 `device_sync`
  (stamped into ATTEND, once per MoE layer; ~20 µs a call would cover it, and a host join was
  measured at 11–20 µs, `how-to-measure.md`); `gemv_fp8` at GLM's `q_a` shape (2048 × 6144),
  1.077× hipcc in arm A (about +0.2 ms/token over 78 layers, also in ATTEND); the Rust attention
  kernels (at parity in arm A). An A/B of B against B-without-the-sync, plus a `gemv_fp8`-only
  arm, would settle it.

**Plan status: COMPLETE** (verdict above). Left open, none blocking: the ATTEND attribution
above *(taken up 2026-11-20 by the follow-up below, which also carries two owed ordering
items)*; the `launchers!` annotations (declined, above); L2's `MoeRange` literals, the shared plant-threshold helper and the plants-root layout
(integration's owed list); CodeScene's 30 inherited files.

### Phase 2 follow-up — the ATTEND attribution (2026-11-20, branch `wave/rk-phase2`, base `372efc6`)

The Phase 2 A/B left B's ATTEND bucket ~1.6 ms/token above A's, unattributed between three
candidates. Logs: `/var/cache/rivoli/scratch/rk-phase2/attr/`.

**`gemv_fp8` at GLM's `q_a` shape, priced from data already on record.** The bit-identity TIME
row `gemv_fp8 (2048, 6144, 128, 1)` read 1.025 / 1.025 (integration A / A′), 1.016 / 1.020
(perf A / A′) and 1.077 (Phase 2 A, one arm, no A′): ~1.02× hipcc, an excess of 0.5–0.8 µs a
call, so 78 calls cost 0.04–0.06 ms/token (0.2 ms at the lone outlier). Not the main cause.
`q_b`, `kv_a` and `o_proj` have no TIME row at GLM's shapes and stay unpriced.

**The lease (2026-11-20 10:01–10:28, rh-anine).** Three binaries, built outside the lock:
A = the pre-migration hipcc build; B = main `372efc6`; S = B with only the JOIN 1
`device_sync()` (and its import) removed, a diagnostic build from `git archive` in its own
target dir, never committed. Order A1 B1 S1 B2 S2 B3 S3 A2, the Phase 2 AB command (512 tokens,
stock flags). Every witness 0 B, GTT 17 MiB before every arm; all eight arms decoded the same
512 ids (md5 `89be6806` over the id lines, header stripped — the Phase 2 AB's too) with the same
236570 hits / 70030 misses. tok/s is 512 ÷ the logged decode seconds; bucket columns are
ms/token from `PROFILE/tok`; host load is the 1-minute loadavg.

| arm | tok/s | attend | ffn | fetch-wait | wall | host load before → after |
|---|---|---|---|---|---|---|
| A1 | 3.088 | 93.750 | 114.242 | 110.102 | 323.760 | 6.72 → 8.11 |
| B1 | 2.754 | 102.509 | 120.313 | 134.587 | 363.166 | 8.11 → 17.06 |
| S1 | 2.671 | 98.137 | 129.013 | 141.323 | 374.457 | 17.06 → 12.18 |
| B2 | 3.055 | 95.979 | 111.855 | 114.100 | 327.365 | 12.18 → 7.43 |
| S2 | 3.035 | 94.373 | 112.623 | 116.947 | 329.453 | 7.43 → 6.09 |
| B3 | 3.060 | 95.695 | 111.403 | 114.122 | 326.667 | 6.09 → 6.94 |
| S3 | 3.092 | 94.308 | 111.537 | 112.198 | 323.490 | 6.94 → 6.15 |
| A2 | 3.097 | 93.838 | 112.873 | 110.453 | 322.821 | 6.15 → 5.40 |

- **Pre-registered verdict: NOT MET.** The rule, fixed by the coordinator before the run: the
  sync explains the gap if mean ATTEND(S) < mean ATTEND(B) − max(spread B, spread S), spread =
  max − min over the three arms. Mean ATTEND B 98.061 (spread 6.814), S 95.606 (spread 3.829);
  Δ 2.455 < 6.814. The spread is pair 1's: host load reached 17 during B1 and S1 (foreign CPU
  load; the GPU witness was clean), inflating every bucket of both arms.
- **Post hoc, outside the rule:** S's ATTEND is below B's in all three pairs (−4.4, −1.6,
  −1.4 ms). In pairs 2 and 3, B averages 95.84 and S 94.34: Δ 1.50 ms/token, ~20 µs a call over
  75 MoE layers, in line with the 11–20 µs host-join price in `how-to-measure.md`; S sits
  0.55 ms above the A anchors (93.79), B 2.04 above. Caveats: S always ran after B, while host
  load fell through pairs 1–2, and only pair 3 is load-matched (−1.39 ms there, against an A1-A2
  difference of 0.09 and a B2-B3 difference of 0.28). The B-vs-A gap itself reproduced (+2.04
  ms in pairs 2–3, +1.6 in the Phase 2 AB).

**The change.** The sync is replaced by the join it restated, as a type:
`DeviceBuf::copy_out_joined` returns a `NullStreamDrained` that only that blocking readback can
mint (`device.rs`, private field), `route_layer` gets it from the gate-logits D2H, and
`launch_moe` takes it. The argument is in the type's doc; the correctness reviewer signed off on
it (the token is exactly as strong as the sync: the sync sat before `submit`, waited on no device
work, and every writer the experts read is on the null stream before the D2H). Taken whatever the
attribution, because by its own argument the sync bought nothing the readback had not.

**Confirmation lease (2026-11-20 10:44–11:04 +01:00, rh-anine).** A = the same `rk-phase2-premig`
binary as above (sha256 `34c58726…`), B′ = the release build of `afedb58` (the token, no sync;
sha256 `de90bedb…`); A1 B′1 A2 B′2 A3 B′3, the Phase 2 AB command. Logs:
`/var/cache/rivoli/scratch/rk-phase2/confirm/`. Every arm rc 0, every witness 0 B, GTT 17 MiB
pre-arm; all six decoded the same 512 ids (md5 `89be6806`, hashed as above) with 236570 hits /
70030 misses. Columns as in the attribution table; means and spreads are computed from the
logged decode seconds, not the rounded column. Pre-registered (coordinator, at GO): the Phase 2
rule (B′ not slower if mean(B′) ≥ mean(A) − max(spread)), and any pair with an arm whose host load
starts or ends above 10 is reported load-confounded, the verdict given with and without it.

| arm | tok/s | attend | ffn | fetch-wait | wall | host load before → after |
|---|---|---|---|---|---|---|
| A1 | 3.040 | 94.606 | 113.407 | 115.191 | 328.895 | 4.16 → 6.00 |
| B′1 | 3.014 | 94.907 | 113.250 | 118.259 | 331.925 | 6.00 → 7.49 |
| A2 | 2.862 | 98.808 | 118.220 | 126.424 | 349.359 | 7.49 → 11.50 |
| B′2 | 3.096 | 94.453 | 111.547 | 111.658 | 323.096 | 11.50 → 5.70 |
| A3 | 3.066 | 94.075 | 112.128 | 114.350 | 326.183 | 5.70 → 4.80 |
| B′3 | 3.088 | 94.376 | 111.333 | 112.731 | 323.858 | 4.80 → 4.33 |

- **All pairs:** mean A 2.989 (spread 0.204), B′ 3.066 (spread 0.082); threshold 2.785, so
  **B′ not slower**. ATTEND B′ − A = −1.25 ms/token, driven by A2's load.
- **Without pair 2** (load-confounded: A2 ends at 11.50, B′2 starts there): mean A 3.053 (spread 0.025), B′ 3.051 (spread 0.075); threshold 2.979, so
  **B′ not slower**. ATTEND B′ − A = +0.30 ms/token (94.64 against 94.34), inside this lease's
  own same-binary spread (A1–A3 and B′1–B′3 both 0.53 ms), so not distinguishable from zero;
  B had sat +2.04 (attribution lease, pairs 2–3) and +1.6 (Phase 2 AB), and S — the same code,
  uncommitted — +0.55.
- **What may be claimed:** decode with the token is not slower than the pre-migration build, and
  the ATTEND excess is no longer measurable above the floor. That the removed sync was the cause
  stays the post-hoc reading; the attribution lease's pre-registered rule was NOT MET.

**Owed (found by the review, pre-existing, not changed here):**
- `stage_batch`'s two uploads (`descs_buf`, `wexpert_buf`) are synchronous `hipMemcpy` H2D from
  pageable memory, issued after the join, and the engine assumes they have landed before the
  expert kernels read them. CUDA's rule lets such a copy return before its DMA lands; ROCm's
  synchronous copy is believed to wait for completion, and nothing here cites it. The removed
  sync sat before `stage_batch` and never covered them (V4 and K3 sync after theirs). Owed: cite
  the ROCm behaviour, or put the uploads behind an event both MoE streams wait on.
  > **CLOSED 2026-11-20 by citation ("stage_batch ordering" below):** ROCm's synchronous
  > `hipMemcpy` does wait for completion, which is both documented and visible in the source. No
  > event was added. The device tripwire is PAID 2026-11-20 ("stage_batch ordering" below).
- The trace-only poison fill inside `submit` (`hipMemsetD32Async` on the null stream) is
  unordered against the fetch write and the miss-stream read of the same slot. Owed: order the
  fill behind the slot's fetch, or argue that the trace's poison may be unordered.
  > **CLOSED 2026-11-23 by citation — it was already ordered.** `routed.rs::admit_misses`
  > (lines 670–684) runs `rivoli_backend::device_sync()` under `trace` after a layer's misses are
  > allocated, i.e. after every poison fill, and before `resolve` hands the reads to the reaper:
  > `alloc` (the only fill site) is called only from `admit_misses`, and `submit` runs
  > `admit_misses` then `resolve`. The review that raised this read `alloc` without its caller.
  > Device evidence: the Phase 0 merge-review lease's trace arm (T) read
  > `read-before-write=0` over 4612 misses.

### stage_batch ordering (2026-11-20, branch `wave/stage-batch`, base `eb801b7`)

**The question.** `stage_batch` (`glm/mlp.rs`) uploads `descs_buf` and `wexpert_buf` with
`DeviceBuf::copy_in_at`, a blocking `hipMemcpy` H2D from pageable `Vec`s on the null stream.
The experts that read them launch afterwards on `compute_stream` and `miss_stream`, both
`hipStreamNonBlocking` (`hiprt::rivoli_stream_create`), which the null stream does not order
against. Nothing else orders the two: the JOIN 1 token comes from the gate-logits D2H, which
runs before `stage_batch`. So correctness depends on one thing: has the copy landed by the time
`hipMemcpy` returns?

**CUDA does not promise it, and CUDA is only context here.** Its API sync behaviour page
(docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html) says: for pageable H2D, "The
function will return once the pageable buffer has been copied to the staging memory for DMA
transfer to device memory, but the DMA to final destination may not have completed."

**HIP does promise it, in its docs and in the code.**
- Docs: the HIP guide's asynchronous-execution page
  (rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/asynchronous.html, fetched
  2026-11-20, HIP 7.17 docs) says "For example, hipMemcpy() for data transfers waits for
  completion before returning control to the host," and "When a synchronous function is called,
  control is not returned to the host thread before the device has completed the requested
  task." The `hipMemcpy` comment in `hip_runtime_api.h` says nothing either way.
- Source: the code is the CLR this box runs. `dev-util/hip-10.0.0` builds
  `libamdhip64.so.7.15.0` from `rocm-clr-10.0.0.tar.gz` (sha256 `07caed9d…`), and none of its
  four patches touches the copy path. `hipMemcpy` → `hipMemcpy_common` → `ihipMemcpy(…,
  stream)`, with defaults `isHostAsync = false, isGPUAsync = true`
  (`hipamd/src/hip_internal.hpp:791`). On the host↔device branch,
  `isHostAsync &= AMD_DIRECT_DISPATCH` (`hip_memory.cpp:766`) leaves it false, so
  `if (!isHostAsync) command->queue()->finishCommand(command)` (`hip_memory.cpp:799`) runs.
  `HostQueue::finishCommand` (`rocclr/platform/commandqueue.cpp:141`) calls
  `IsHwEventReady(event, kWaitCompletion = true)`, which waits on the command's HSA signal
  (`rocdevice.cpp:3207–3221`), or falls back to `awaitCompletion()` if the command has no HW
  event.
- Which copy runs: `blocking_blit_ = IS_HIP`, but `hipMalloc` memory is not
  `HostMemoryDirectAccess`, so the CPU-memcpy path is not taken. Two things rule it out:
  `HSA_ENABLE_DTIF_FAST_COPY` defaults false, and the fine-grain allocation that would also get
  direct access only applies on a FULL-profile agent (`rocmemory.cpp:898–906`). The paths, from
  `rocsettings.cpp` and `rocblit.cpp`, are:
  - up to `GPU_FORCE_BLIT_COPY_SIZE` (16 KiB): a staging buffer and a shader copy that carries a
    completion signal and a system-scope release (`KernelBlitManager::writeBuffer`,
    `rocblit.cpp:2255`);
  - up to `pinnedMinXferSize_` (64 KiB): staged SDMA;
  - above that: SDMA from the pageable source pinned in place, in 32 MiB chunks
    (`DmaBlitManager::getBuffer`, `rocblit.cpp:1042`).

  Both real uploads take the first path: 36–136 B for `wexpert` and 432/816 B for `descs` at
  nrow 1/2. Since `slots = top_k·MAXROW + n_shared`, they stay there for any `top_k ≤ 170`.

**Why that is enough, for the shader-copy path.** The host returns only after the copy has
completed, and only then enqueues the expert kernels. The copy is a kernel on the same agent,
so it shares the L2 with the experts, and it releases at system scope. The null stream's lack of
ordering against non-blocking streams never matters, because nothing is left in flight on it
when the experts are enqueued. For the SDMA paths (above 16 KiB), the reader's acquire is
agent-scope (`AMD_OPT_FLUSH`), and the hardware ordering of an SDMA write against it is NOT
argued here. The tripwire covers those sizes empirically, and `stage_batch` never uses them.
`copy_in_at`'s doc carries the citation and this scope, and `stage_batch` points to it.
Correctness review signed off on the argument for `stage_batch`'s sizes and asked for this scope.

**The tripwire.** `crates/engine/tests/h2d_ordering.rs`, rocm arm. Each iteration uploads fresh
data with `copy_in_at`, then immediately folds it with `hash_rows` on a `HipStream::miss()`
stream with no other sync. A device fold that differs from `rivoli_core::hash::xor_fold` means a
stale read. Sizes are 36, 432 and 816 B (the real uploads), then 16 KiB, 64 KiB, each of those
plus 4, then 1 MiB and 16 MiB, with 16 to 256 iterations per size.

The red proof is a standing fixture that runs every time: `async_upload_without_a_wait_is_seen_stale`.
It is the same loop with the upload replaced by a 64 MiB `hash_rows` delay kernel and a
`hipMemcpyAsync` from pinned memory, both on the null stream, and no wait. It must see stale
reads at EVERY size. The delay is what gives it power at the real sub-KiB sizes, because without
it a small copy finishes before the reader starts. If a small size reads 0 stale even so (for
example, if HIP placed both streams on one hardware queue, whose barrier bit would serialise
them), the fixture fails and says the tripwire has no power at that size. Device arm: **PAID**
(see the dated note below).
(READY-FOR-GPU).

**Decode cost.** Nothing in the decode path changed, so there is no decode A/B to run.

### CodeScene and stage_batch lease (2026-11-20, branch `wave/cs-integrate` @ `d6eb521`)

**CodeScene 1.0.58 green.** Thirty pre-existing files scored below 10 under CLI 1.0.58 (18 of
46 issues "Excess Number of Function Arguments"). Four lanes refactored them structure-only —
named parameter structs and newtypes (tuple-packing to dodge the argument count was refused as
hiding arity), phase-split constructors, flattened test nests, the V4 oracle's compressor and
indexer split into helpers — with no threshold or config change and no new EXEMPT row. Converter
output byte-identical (312 files across six converter suites), V4 oracle goldens and the qwen
anchor's 80 goldens byte-identical, an independent correctness read of the oracle refactor clean.
`codescene.rs` passes on the integrated tree (deviceless arm 517 passed, 0 failed; the gate
scored the whole tree in 253 s and its red-proof fixture stayed below 10).

**Device lease** (coordinator-run, 17:59–18:49 +01:00; every witness 0 B, GTT 17.8 MiB before and
after D and P, nothing compiled inside the lock):

| arm | result |
|---|---|
| H, `h2d_ordering` | blocking pageable upload 0 stale reads at all nine sizes (36 B … 16 MiB, 16–256 iterations each); the standing red fixture stale on every iteration at every size (256/256, 16/16) — the tripwire has power and the citation holds on this device |
| A, bit-identity | GREEN, every row BIT-IDENTICAL, barrier census SAME |
| W, full battery | 724 passed, 0 failed, 2 ignored — the first fully green battery (CodeScene included); every refactored suite ran |
| D, determinism (`--arena-refresh`, 1024) | GREEN, identical ids to 1024 |
| P, parity-glm | GREEN, 32 generated ids identical to the pinned reference |

GLM MoE is correct after the refactors and the stage_batch citation: parity, determinism and
the battery's GLM suites all green on the integrated tree.

### Open items closed (2026-11-23, branch `wave/open-items`)

- **The trace poison fill** — closed by citation; see its owed bullet above.
- **The `/tmp` scratch defaults.** Eight gate scripts (`determinism-glm`, `device-halves`,
  `feature-matrix`, `nll-divergence`, `parity-glm`, `ppl-gates`, `smoke-glm`, `smoke-v4`) spelled
  `mktemp -d "${TMPDIR:-/tmp}/…"`, so an unset `TMPDIR` put id dumps and witness files in the RAM
  tmpfs CLAUDE.md forbids. `tests/scratch.sh` (`scratch_root`, `scratch_dir`) now chooses: an
  explicit `TMPDIR`, else `/var/cache/rivoli/scratch` when it exists, else `/tmp` (CI's hosted
  runner only). `determinism-glm.sh --self-test` passes through it.
- **The integration owed list** — `MoeRange`, the shared plant bar, the gdn `rep`, the plants
  layout: see the dated block under that list.
- **The ATTEND attribution** — re-run with a counterbalanced order and a load gate, pre-registered
  in `scratch/open-items/attr/attr.sh`; result below.

**ATTEND attribution, second attempt: INCONCLUSIVE (2026-11-23).** Pre-registered in
`scratch/open-items/attr/attr.sh` before any arm ran: B = main `995ceb3` (JOIN 1 as the token),
S = B with the pre-`afedb58` `device_sync()` restored inside the ATTEND lap; counterbalanced order
B1 S1 S2 B2 B3 S3 S4 B4 B5 S5 (the first attempt ran S always after B on falling load); an arm is
VALID only with rc 0, an empty witness and host 1-min loadavg ≤ 8.0 before and after; a verdict
needs ≥ 3 valid arms of each binary. Every witness was 0 B, GTT 17.8 MiB before the lease (after
llama-swap's embedding model was unloaded).

| arm | load before → after | attend ms/tok | wall ms/tok | valid |
|---|---|---|---|---|
| B1 | 6.35 → 7.70 | 97.624 | 356.6 | yes |
| S1 | 7.70 → 7.75 | 96.864 | 354.6 | yes |
| S2 | 7.75 → 8.87 | 96.273 | 351.2 | no |
| B2 | 8.87 → 9.11 | 101.426 | 393.1 | no |
| B3 | 9.11 → 7.60 | 97.683 | 349.3 | no |
| S3 | 7.60 → 6.14 | 97.359 | 337.7 | yes |
| S4 | 6.14 → 12.53 | 99.757 | 354.2 | no |
| B4 | 12.53 → 8.39 | 96.913 | 362.2 | no |
| B5 | 8.39 → 8.84 | 159.746 | 674.5 | no |
| S5 | 8.84 → 7.30 | 115.941 | 442.9 | no |

Valid: B 1 arm, S 2 arms — below the pre-registered 3, so **INCONCLUSIVE**, and no arm is rerun.
The host was not quiet: a user `valgrind` memcheck job (one core, ~3 days) and periodic root
`nuq-worker` node processes pushed the 1-min load from 6 to 12.5 during the lease. Descriptively,
on the three valid arms S's ATTEND (96.9, 97.4) sits at or below B's (97.6), the opposite sign
from the post-hoc reading — but three arms establish nothing, and it is recorded as data, not a
verdict. **The standing claim is unchanged:** decode with the token is not slower than the
pre-migration build (two leases), and the ATTEND excess is no longer measurable above the floor;
the sync-as-cause reading stays post-hoc. A third attempt needs a quiet host (no co-tenant CPU
load) — the method in `attr.sh` is ready to re-run as is.

*Observation, not a finding:* S2 and S5 (both S) decoded different ids from the other eight arms,
first diverging at tokens 176 and 325; all five B arms were identical. That is GLM's known
stock-flags nondeterminism (`glm-nondeterminism-closeout.md`; no `--arena-refresh` here, and its
rate rises with load). 2/5 against 0/5 is not significant (Fisher exact p ≈ 0.44).
