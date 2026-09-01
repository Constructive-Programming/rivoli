---
status: live
scope: engine
verdict: ROCm 10.0.0 is INSTALLED, WORKING and MEASURED on rh-anine as of 2026-09-01, and the migration's open column is closed: after the overlay's hip was rebuilt with `-mno-avx` (the device fault was an alignment #GP — `-march=znver5` emitted 64-byte-aligned stores into a `roc::VirtualGPU` that clr's own class-scope `operator new(size_t)` heap-allocates 16-byte aligned, latent UB invisible in AMD's generic-x86-64 prebuilt; both suspects named by attempt 2, the unpackaged `librocm_kpack` and the LLVM link mode, are REFUTED), all seven cells of the pre-registered re-book spec are GREEN in a fresh target dir with witness-clean arms and tenants parked at the baseline's 17 MiB GTT floor: device battery 91 suites / 592 tests / 0 failed with zero `Compiling` inside the lock, pinned V4 decode x2 BYTE-IDENTICAL to each other AND to the 7.14 column (raw sha 431606e9, id-lines 67233d64), GLM A-vs-A 512/512 identical at the baseline's 100 GiB budget, the GLM bench cell's recorded ids, and tests/parity-glm.sh 32/32 identical to the pinned reference at full reference length. Numerics did not move — same goldens, same tolerances, and hit/miss counts equal TO THE DIGIT in every arm across the toolchain change. Throughput is unchanged within a percent and straddles zero (6.32/6.43 vs 6.32/6.40; 2.91/2.88 vs 2.93/2.94), so NO throughput claim is made. Under the pre-registered decision rule there is no red and nothing to roll back; the 7.14 binpkgs stay armed as the 43 s escape. Four conditions travel with the column, including that cells 2-3 ran at loadavg 10.9-12.9 (a sibling agent's CPU work, no GPU contact) and that parity-glm.sh exits 1 on a USAGE error, which is its codebook's value for a token mismatch.
---

# ROCm 7.14 → 10.0.0 on rh-anine — the measured migration

**Why bump** (plan B0): gfx1151 is an officially listed supported target for the first time
in ROCm 10.0.0 (Ryzen AI Max+ 395 in the compatibility matrix; 7.x never listed it), the
release claims improved roofline profiling for gfx1150/1151/1152, and the qwen port's new
kernels get authored once against the new HIP instead of ported later.

**What "10.0.0" actually is** (hr-fleet feasibility, 2026-08-30, verified against AMD's own
shipped artifacts, not its docs): AMD's compatibility matrix says LLVM 24.0.0 + HIP 10.0.0;
the shipped `therock-dist-linux-gfx1151-10.0.0.tar.gz` (1,794,376,103 B) carries
**clang 23** (`lib/llvm/lib/clang/23`), **HIP 7.15.26333** (`libamdhip64.so.7.15.26333`),
and `libamd_comgr.so.3.3.0` / `libhsa-runtime64.so.1.21.0` **byte-identical version strings
to what 7.14 installed**. The SONAME is unchanged (`libamdhip64.so.7`, read from the real
library with `readelf -d`, with a negative control at tag `therock-7.14` predicting the
installed `7.14.60850` exactly). So the real delta is ONE HIP minor on an unchanged
compiler, and the throughput expectation was set to modest-or-nothing BEFORE the A/B.
Do not cite the matrix's version numbers; cite the binary.

## Toolchain pins

| | OLD (baseline) | NEW (after merge) |
|---|---|---|
| HIP | 7.14.60850-0000000 | not measured — attempt 1 rolled back (see Outcome) |
| clang | 23.1.0 (`/usr/lib/llvm/23`) | not measured — same |
| kernel | 6.18.41-gentoo (in-kernel amdgpu, no driver phase) | same (no reboot planned) |
| `libamdhip64` SONAME | `.so.7` | not measured (expected `.so.7`, confirmed from the shipped tarball) |
| package set | dev-util/hip-7.14.0, hipcc-7.14.0, rocm-smi-7.14.0, rocminfo-7.14.0, dev-libs/rocm-{core,comgr,device-libs}-7.14.0, rocr-runtime-7.14.0, roct-thunk-interface-7.14.0, rocm-cmake-7.1.0 (+ rocprofiler-therock-bin) | not merged — rolled back to the 7.14 set (ten 10.0.0 atoms planned; rocprofiler-register stays 0.6.0 — ROCm 10 ships the identical version, all consumers dep on it unversioned) |

**Install and rollback are hr-fleet's** (owner decision 2026-08-30): the rhansen overlay's
TheRock-based ebuild set, bumped via its single `ROCM_TAG` anchor to `therock-10.0`; plan at
`hr-fleet:docs/plans/2026-08-30-001-feat-rocm-10-rh-anine-plan.md`. Rollback did not exist
until this migration forced it: `quickpkg` of all 11 current atoms into `/var/cache/binpkgs`
(12 binpkgs, 52 MB), gated on `emerge --usepkgonly --pretend` resolving every atom
`[binary R]`, overlay tagged `rocm-7.14.0-known-good`. Rollback is one
`emerge --usepkgonly` (~15 min) — plus a rivoli rebuild, because the kernels are compiled
by whichever hipcc is installed; that asymmetry is budgeted on both exit paths.
The `~amd64` keyword file stays OUT of `/etc/portage` until inside the window: rh-anine's
nightly 00:00 autonomous `@world` run would otherwise merge ROCm 10 unattended.

## Window state (both columns run inside it)

GPU tenants parked by hr-fleet at window open 2026-08-30 ~23:30Z: `hr-home.xyz/rocm` label
flipped to the VALUE `disabled` (the device-plugin DaemonSet reconciled to 0), and — because
a label flip never evicts a running pod, `nodeSelector` is scheduling-time only — llama-swap
parked by fleet-bundle pause + scale-to-0. `mem_info_gtt_used` 7,839,670,272 B → 18,673,664 B;
`/sys/class/kfd/kfd/proc/` empty (it was ALSO empty with 7.8 GB held — KFD is blind to
Vulkan tenants, which is why the witness samples GTT). Restore is two steps + bundle
unpause LAST, after enumerating anything Renovate merged against `fleet/ai/**` during the
window (the 2026-08-13 flood scar).

## Baseline — OLD toolchain (all arms 2026-08-30/31, witness clean on every device arm)

Full manifest with per-arm logs, unpiped `.exit` files and witness files:
`/var/cache/rivoli/scratch/rocm10-migration/baseline/MANIFEST.md` (rh-anine). Build in a
FRESH `CARGO_TARGET_DIR=/var/cache/rivoli/target/rocm10-mig-old` (never shared across
toolchains — §7e-bis/§12 class), 15/15 HIP TUs to `*.gfx1151.o`, CLI links
`libamdhip64.so.7`.

| arm | result | key numbers |
|---|---|---|
| device battery (flocked, threads=1, dev profile) | exit 0, witness CLEAN | 89/89 suites, **576 passed / 0 failed**, 17 min 01 s, 0 `Compiling` lines inside the lock |
| pinned decode ×2 (V4 fp4, `--attn dsa --max-mem 100 --ctx 2048 --bench 32`, no build between runs) | exit 0 ×2, witness CLEAN ×2 | **byte-identical** (both raw sha `431606e9…`) AND equal to the committed `tests/v4-bench32-ctx2048.ids`; **6.32 / 6.40 tok/s** |
| GLM A-vs-A (`tests/determinism-glm.sh`, release, 512 tokens, **100 GiB budget**) | exit 0, witness CLEAN, self-test red proof ran first | **512/512 ids identical** (both arms sha `29c9fbd6…`), 2.93 / 2.94 tok/s, identical 223148 hits / 83452 misses |
| GLM bench cell (int3-vq, `--bench 4`) | exit 0, witness CLEAN | ids `[13041 1052 0 358]` == recorded reference; throughput NOT citeable (n=4, warm cache) |

Three caveats that travel with these numbers:
1. **GLM was byte-identical this pair.** The open streaming-nondeterminism defect did not
   reproduce (the gate's Poisson bound gives ~31–99% power at 512 tokens). A red on the
   ROCm 10 column is therefore NOT a regression by itself — it is the standing defect
   firing; judge the new column against the defect's record, not against this green.
2. **The A-vs-A ran at a 100 GiB budget, not the script's 115 GiB default**
   (MemAvailable was 110.7 GiB; the pin is system RAM and a live peer service was
   protected). The ROCm 10 column must run at the SAME budget or the rows do not compare.
3. The "pinned decode" arm is DeepSeek-V4 fp4 (the `v4-bench32-ctx2048.ids` cell) — it is
   the byte-identity instrument here; the Muse-Glimmer bf16 fully-pinned cell was not
   re-taken this window. Absolute tok/s everywhere is conditioned on standing host CPU
   tenants (loadavg 2.4–4.7 during arms; recorded in `00-host-tenants.txt`).

## ROCm 10 column — not measured (attempt 1 rolled back); the protocol below is the re-book spec

When the re-booked merge reports green: rebuild EVERYTHING in a second fresh dir
(`/var/cache/rivoli/target/rocm10-mig-new`); deviceless suite; flocked battery against the
SAME vendored goldens at the SAME tolerances (a trip means codegen moved numerics past a
measured floor — investigate, never widen); pinned decode ×2 (byte-identity is THE
toolchain-determinism instrument — the baseline column proves it holds under 7.14, so a
divergence here is the toolchain's); GLM A-vs-A at 100 GiB; `tests/parity-glm.sh` full run;
bench A/B against the numbers above.

**Decision rule** (pre-registered): any red that is not a measured-and-accepted improvement
rolls back — one message to hr-fleet, `emerge --usepkgonly`, rivoli rebuilt under 7.14, and
this doc's verdict records the rollback and why.

## Outcome — attempt 1 (2026-08-31): ROLLED BACK, re-book gated on three offline fixes

The merge ran in two attempts inside the window. Attempt 1 failed at 5/10 atoms on the
msad-target-feature patch, which AMD had upstreamed into therock-10.0 (hunk 2 refused;
the fix was deleting the patch line — note the ebuild spells it `${PN}-…`, so the first
recorded sed pattern matched nothing). Attempt 2 cleared the patch layer in 78 s and
exposed three build-layer blockers, all deviceless to fix:

1. **rocr-runtime**: `hsakmt/linux/kfd_ioctl.h` not found — the header exists in both
   source tarballs but `roct-thunk-interface` never installs it; therock-10.0's new
   `amd_core_dump.cpp` includes it through the household `use-system-hsakmt` patch.
2. **rocm-comgr**: `-lLLVMBinaryFormat` (+9 more) unresolvable — the installed LLVM 23 is
   dylib-only (`libamd_comgr.so.3.3.0` NEEDs `libLLVM.so.23.1`), and the ebuild does not
   pass the CMake link-mode for it.
3. **hipcc-10.0.0**: invokes the unslotted `/usr/lib/llvm/bin/clang++`, which does not
   exist on Gentoo — live breakage while installed (worked only under HIP_CLANG_PATH).

Rollback: `emerge --usepkgonly` over the 11 binpkgs, **43 s, zero compilation**; the
restored `libamdhip64.so.7.14.60850` carries its original Aug 14 mtime (the binpkg's
bytes, not a rebuild); keyword file removed and `emerge -uDNp @world` re-verified showing
zero ROCm lines; overlay auto-sync restored. Post-rollback verification on the rivoli
side: the pinned decode cell (arm-3 form, run once via the flocked+witnessed `run_arm`)
reproduced `431606e9…` — byte-identical to baseline `03-r1.ids` — witness 0 bytes.

Also recorded from the window: the nightly `emerge --sync` had DELETED the ten
uncommitted overlay ebuilds (`reset --hard` on untracked-adjacent state) — everything
deliberately parked outside the overlay (binpkgs, tag, staged keyword file, seeded
distfile) survived; the recreated ebuilds are committed and the branch pushed upstream.
Re-book gate: all three fixes above build green via `ebuild … compile` in an offline
session; the window shape then shrinks to merge + rivoli re-run, since this baseline
column stays valid while the 7.14 stack is byte-identical.

## Attempt 2 (2026-08-31, owner-installed): compiles, deviceless green, DEVICE RUNTIME BROKEN

Installed set verified: nine atoms at 10.0.0 (hip, hipcc, comgr, core, device-libs, rocr,
roct, rocm-smi, rocminfo), `hipcc` reports HIP 7.15.0 / clang 23.1.0, SONAME `.so.7` as the
tarball predicted. **Not installed:** `dev-util/rocprofiler-therock-bin` 10.0.0 (no rocprof*
binaries — the gfx1151 roofline tooling was the bump's headline payload) and
`rocprofiler-register` no longer lists in qlist (its 0.6.0 files remain on disk, orphaned).

Measured, in order, all commands and per-arm logs under
`/var/cache/rivoli/scratch/rocm10-migration/new/` on rh-anine:
1. **Compile: GREEN.** Fresh `CARGO_TARGET_DIR=/var/cache/rivoli/target/rocm10-mig-new`,
   `cargo build --workspace` exit 0, 117 crates, 15/15 HIP TUs to `*.gfx1151.o`, CLI links
   `libamdhip64.so.7`.
2. **Deviceless suite: GREEN** (91 suites, 0 failed).
3. **Device battery: RED.** rc=101 at suite 33 — `rivoli-backend --lib`,
   `gpustream::tests::signal_resolves_and_latency`, **SIGSEGV**. Witness clean, GTT 17 MiB
   pre-arm. Reproduced 2 more times in isolation, instantly, deterministically.
4. **Fault located.** `AMD_LOG_LEVEL=3`: runtime init completes (ROCr backend initialized,
   GPU enumerated), then the FIRST HIP call — `hipStreamCreateWithFlags(&s,
   hipStreamNonBlocking)` — never returns. Coredump backtrace: five frames inside
   `libamdhip64.so.7`, entered from `hipStreamCreateWithFlags` ←
   `rivoli_stream_create` ← `HipStream::new` (gpustream.rs:96) — the same call the 7.14
   battery passed 592/592 with hours earlier on identical rivoli code.
5. **The A/B that convicts the build, not the release:** the SAME test binary under
   `LD_LIBRARY_PATH` = AMD's own prebuilt tarball `lib/` (hip 7.15.26333, hsa 1.21.0)
   **passes, 0.37 s, exit 0**, flocked + witnessed. One-lib bisect: prebuilt hsa alone and
   prebuilt rocprofiler-register alone still segfault (the defect is not theirs); prebuilt
   hip cannot be swapped alone because it NEEDs **`librocm_kpack.so.0`** — a NEW ROCm 10
   component the overlay does not package at all — and prebuilt comgr NEEDs the tarball's
   own `libclang-cpp.so.23.0git`, so the hip/comgr/LLVM trio only runs as a set.
6. **Verdict:** the overlay-built `dev-util/hip-10.0.0` (built against the dylib-only
   system LLVM 23, without rocm_kpack — i.e. exactly where attempt 1's blockers B and C
   were hand-fixed) is defective at stream creation. ROCm 10 itself is fine on this
   kernel; rivoli is fine.

**Paths from here (owner's call):** (a) rebuild `dev-util/hip` to TheRock's own
configuration (package `rocm_kpack`, match AMD's LLVM link mode) — hr-fleet's lane;
(b) install AMD's prebuilt dist as the runtime (the tarball already validates on this box);
(c) roll back to the 7.14 binpkgs (43 s, proven) until (a) lands. Until one of these,
EVERY rivoli device arm on this box faults at engine start — batteries, anchors' GPU
sessions, S7 — while builds and deviceless suites keep working.

## 2026-09-01: the fault reproduces WITHOUT rivoli, and (a) is the owner's chosen path

Attempt 2's conviction rested on a rivoli test binary. It does not need one. A four-line
translation unit — `hipStreamCreateWithFlags(&s, hipStreamNonBlocking)` and a `printf` of
the returned `hipError_t`, nothing else, compiled by the installed
`hipcc --offload-arch=gfx1151` — behaves as follows on this box today, both arms flocked:

| runtime | result |
|---|---|
| installed overlay `libamdhip64.so.7.15.0-0000000` | **exit 139 (SIGSEGV)**, no line printed — the call never returns |
| AMD prebuilt `lib/` + `prebuilt-libs/lib/` under `LD_LIBRARY_PATH` | `hipStreamCreateWithFlags rc=0`, **exit 0** |

Probe and both invocations: `/var/cache/rivoli/scratch/hipprobe.hip`, built to
`/var/cache/rivoli/scratch/hipprobe`. The prebuilt arm needs **both** directories on the
path: `librocm_kpack.so.0` lives in `prebuilt-libs/lib`, and with only `rocm10/lib` the
loader fails at exit 127 before the GPU is touched at all — which is a different failure
from the one under test, and reading 127 as "the prebuilt is broken too" is the trap here.

This removes rivoli, its kernels, its io_uring rings and its arena from the claim entirely:
the defect is in the overlay HIP build, at the first HIP call any program makes. It also
narrows the fix — `librocm_kpack.so.0` is a hard `DT_NEEDED` of AMD's own hip, and the
overlay packages no such library.

**Owner's call 2026-09-01: path (a).** `dev-util/hip` is rebuilt to TheRock's own
configuration (package `rocm_kpack`, match AMD's LLVM link mode) in hr-fleet's lane; (b) is
NOT taken as an install, and (c) stays armed as the 43 s escape. rivoli device arms stay
blocked until that rebuild lands, and the ROCm 10 column below stays not-measured. The
probe above is the acceptance test for the rebuild: it must print `rc=0` against the
INSTALLED library, with nothing on `LD_LIBRARY_PATH`.

## 2026-09-01, later the same day: the blocker is FIXED, and it was neither suspect

> **CORRECTS the section above and attempt 2's paths list.** "Path (a) is chosen … rivoli device
> arms stay blocked until that rebuild lands" was true for about four hours. The rebuild landed.
> The two suspects that section named — the unpackaged `librocm_kpack` and the LLVM link mode —
> are both **refuted**, so the sentence "it also narrows the fix" narrowed it toward the wrong
> thing and is corrected here rather than deleted.

**The defect** (diagnosed in hr-fleet's lane; its record is that repo's
`docs/plans/2026-08-30-001-feat-rocm-10-rh-anine-plan.md` PART IX §43–46, the fix is overlay
commit `001a4b9`): an **alignment #GP**, not a bad pointer. The faulting instruction is
`vmovdqa64 %zmm1,0xc0(%rdi)` inside `roc::VirtualGPU`'s constructor with `rdi` 16-byte aligned.
`VirtualGPU` declares two `alignas(64)` AQL packets, but is heap-allocated through clr's
class-scope `amd::ReferenceCountedObject::operator new(size_t)`, which **hides C++17's
`::operator new(size_t, align_val_t)`** — the compiler proved `alignof == 64` and emitted aligned
stores, `new` returned 16. Latent UB in AMD's own source, reachable only because this overlay
builds with `-march=znver5`; AMD's `therock-dist` builds generic x86-64 and emits no `%zmm` at
all, which is why the same source version works there. The fix is one line in the hip ebuild,
`append-flags -mno-avx`, chosen over patching the header because four more clr classes have the
identical hole. No revbump, deliberately: the keyword file pins `=dev-util/hip-10.0.0` exactly,
so an `-r1` would be masked and would arm a nightly `@world` **downgrade**.

**Why the kpack evidence was noise, written down because it fooled attempt 2's write-up.**
`LD_PRELOAD`ing kpack made the probe pass — and so did `env A=x` with no preload at all, while
kpack + comgr together went back to 139 and a 34-byte environment variable failed. Three bytes of
environment flipped it either way: that is a heap-layout coin toss, and mechanically a missing
`DT_NEEDED` is a load-time failure, not a mid-constructor #GP. `librocm_kpack` was NOT packaged;
nothing in this tree links it.

**Verified in this tree, independently of that diagnosis** (the point of re-deriving rather than
relaying: the fix's own author is the last party who should score it):

| check | result |
|---|---|
| four-line probe, recompiled with the new `hipcc`, `env -u LD_LIBRARY_PATH -u LD_PRELOAD`, flocked | `hipStreamCreateWithFlags rc=0`, **exit 0** (exit read unpiped) |
| `objdump -d` of the installed `libamdhip64.so.7.15.0-0000000` | **0** `%zmm`, **0** `vmovdqa`/`vmovaps` — matching what AMD's prebuilt has |
| `cargo test -p rivoli-backend --lib -- --test-threads=1`, flocked, dev profile | **4 passed / 0 failed** in 0.22 s, including `gpustream::tests::signal_resolves_and_latency` — the exact test that SIGSEGV'd three times |
| rebuild needed? | **no** — the workspace build was already fresh (0.11 s, no `Compiling`), SONAME and `DT_NEEDED` unchanged by the flag |

**What is still owed: the ROCm 10 column itself.** The four cells the baseline table holds
(device battery, pinned V4 decode ×2, GLM A-vs-A at the **100 GiB** budget the baseline used, GLM
bench) need the same tenant-parked window the baseline ran in. The witness on the arm above was
**not clean**: `mem_info_gtt_used` read 7,839,670,272 B before it and 7,839,715,328 B after, a
live Vulkan tenant (llama-swap, invisible to KFD) that the baseline did not have. That does not
weaken a functional green — a stream either creates or faults — but it disqualifies every
throughput number, so none is quoted here. The column is booked, not measured.

## ROCm 10 column — MEASURED 2026-09-01, every cell green

> **SUPERSEDES the "not measured" heading above**, which was written for attempt 1's rollback
> and stayed true through attempt 2's broken runtime. The protocol in that section is the spec
> this column was run against; the spec itself is unchanged and is not restated here.

Run after the hip rebuild, in a **fresh** `CARGO_TARGET_DIR=/var/cache/rivoli/target/rocm10-mig-new2`
— fresh because the hip PACKAGE was reinstalled between attempt 2's build and this one, which is
the §7e-bis/§12 class the spec's own rule guards. Driver, every log, every `.ids` file and every
per-arm witness: `/var/cache/rivoli/scratch/rocm10-migration/new2/` on rh-anine; the driver is
`/var/cache/rivoli/scratch/rocm10-column2.sh`.

**Window conditions.** GPU tenants parked by hr-fleet at 16:2xZ — node label `hr-home.xyz/rocm`
flipped to the value `disabled`, `llama-swap-rh-anine` scaled to 0 and its fleet bundle paused.
`mem_info_gtt_used` 7,839,670,272 B → **18,673,664 B**, the same idle floor the 7.14 baseline ran
against, verified here rather than relayed. Two deviations from the recorded park procedure are
worth carrying: the Deployment was **renamed** by `e5f6c1bc` (`ai/llama-swap` no longer exists; it
is now `llama-swap-rh-anine` for Vulkan and `llama-swap-hr-main` for CUDA), so the recorded command
would have failed NotFound and read as "nothing to park"; and the manifest now carries an explicit
`replicas: 1`, so the bundle pause is load-bearing in a way it was not on 08-30.

| # | cell | exact form | exit | witness | result vs the 7.14 baseline |
|---|---|---|---|---|---|
| 0 | probe | four-line `hipStreamCreateWithFlags`, env stripped, flocked | 0 | n/a | `rc=0` against the INSTALLED library |
| 1 | deviceless suite | `cargo test --workspace --no-default-features` | 0 | n/a | **91 suites, 0 failed** |
| 2 | builds, OUTSIDE the lock | test bins, `--example glm_smoke`, `--release` | 0, 0, 0 | n/a | all three link `libamdhip64.so.7` |
| 3 | **device battery** | flocked, `--test-threads=1`, dev profile | **0** | **CLEAN** (0 bytes) | **91 suites, 592 passed, 0 failed**; 17 min; **0 `Compiling` inside the lock**; GTT 17 MiB pre-arm. Baseline: 89 suites / 576 — the tree grew, the goldens and tolerances did not |
| 4 | **pinned V4 decode ×2** | `--attn dsa --max-mem 100 --ctx 2048 --bench 32`, debug bin, **no build between** | 0, 0 | **CLEAN** ×2 | **BYTE-IDENTICAL to each other AND to the 7.14 column**: raw sha `431606e9…` on r1, r2 and the baseline's `03-r1.ids`; id-lines `67233d64…`. **6.32 / 6.43 tok/s** (baseline 6.32 / 6.40); **6217 hits / 1781 misses in all four runs** |
| 5 | **GLM A-vs-A, 512 tokens, 100 GiB** | `determinism-glm.sh --self-test` first, then the pair | 0 | **CLEAN** ×2 | **512/512 ids IDENTICAL**; 2.91 / 2.88 tok/s (baseline 2.93 / 2.94); **223148 hits / 83452 misses in both arms and in both baseline arms** |
| 6 | GLM bench cell | `--mode int3-vq --attn dense --max-mem 100 --bench 4`, debug bin | 0 | CLEAN | ids `[13041 1052 0 358]` == the recorded reference; 3.77 tok/s over 4 tokens — **not citeable** (n=4, cache warm from cell 5) |
| 7 | **`tests/parity-glm.sh`** | `glm52-vq3-full 32 100`, never builds | **0** | **CLEAN** ×2 | **32 generated ids IDENTICAL to the pinned reference at reference full length** |

**Verdict against the pre-registered decision rule: no red, nothing to roll back.** The rule said any
red that is not a measured-and-accepted improvement rolls back to the 7.14 binpkgs. There is no red.
Numerics did not move: the same vendored goldens at the same tolerances, and the two id-identity
instruments (pinned decode, parity) are byte-equal ACROSS the toolchain change, not merely
self-consistent within it.

**Throughput: no change worth the name, as pre-registered.** The bump's expectation was set to
modest-or-nothing before the A/B, and the measurement agrees — 6.32/6.43 against 6.32/6.40, and
2.91/2.88 against 2.93/2.94, i.e. within a percent and straddling zero in both directions. **No
throughput claim is made from this column.** The hit/miss counts being identical to the digit in
every arm is the stronger statement: the streaming path executed the same work, not merely a
similar amount of it.

Four conditions this column carries, recorded rather than smoothed:

1. **Cells 2 and 3 ran at loadavg 10.9–12.9, not the baseline's 2.4–4.7.** A sibling agent's
   deviceless `cargo test --workspace` was on 32 cores at the time and said so. It never touched
   `/dev/kfd` or GTT, so no contention witness is affected and the battery's colour is unaffected —
   but had a throughput number come off those two cells it would have been uncitable. Cells 4–7 ran
   at 3.2–4.9, inside the baseline's own band.
2. **Cell 5 is not evidence about the standing GLM streaming-nondeterminism defect.** The 7.14
   column was byte-identical here too; the gate's Poisson bound gives ~31–99% power at 512 tokens.
   Both columns green means the defect did not fire in either, which is what the baseline's caveat 1
   already said.
3. **Cell 7 ran at the gate's default `ngen=32`, not a longer window.** That is the gate as it is
   defined, and the 7.14 baseline recorded no parity arm at all, so this row is a new absolute
   result rather than a comparison.
4. **A defect in the gate's own error path, found by tripping it.** `parity-glm.sh` was first
   invoked with no artifact argument; it printed its usage line and exited **1** — and 1 is its
   codebook's value for *gate RED, token mismatch*, while setup errors are supposed to exit 2. A
   wrapper classifying on the exit code alone would have recorded a parity failure under ROCm 10
   that never ran. The run in the table is the re-invocation with the argument.
