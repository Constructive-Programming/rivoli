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
  `hipMemMap`'d pool slot is unverified on this ROCm; `attn_res`'s private `p[16]` lands in
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
  64/64 seeds). Device arms: pending the GPU lease.

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
