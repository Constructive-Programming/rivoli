---
status: live
scope: engine
verdict: ROCm 7.14 -> 10.0.0 migration IN FLIGHT — the old-toolchain baseline is complete and green on all five arms (battery 576/576, V4 decode byte-identical to the pinned reference, GLM A-vs-A 512/512 identical at a 100 GiB budget), the merge is delegated to hr-fleet with a proven binpkg rollback, and the ROCm 10 column is not yet measured; no number here transfers to the new toolchain until it is.
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
| HIP | 7.14.60850-0000000 | RESULT-PENDING |
| clang | 23.1.0 (`/usr/lib/llvm/23`) | RESULT-PENDING |
| kernel | 6.18.41-gentoo (in-kernel amdgpu, no driver phase) | same (no reboot planned) |
| `libamdhip64` SONAME | `.so.7` | RESULT-PENDING (expected `.so.7`) |
| package set | dev-util/hip-7.14.0, hipcc-7.14.0, rocm-smi-7.14.0, rocminfo-7.14.0, dev-libs/rocm-{core,comgr,device-libs}-7.14.0, rocr-runtime-7.14.0, roct-thunk-interface-7.14.0, rocm-cmake-7.1.0 (+ rocprofiler-therock-bin) | RESULT-PENDING (ten 10.0.0 atoms; rocprofiler-register stays 0.6.0 — ROCm 10 ships the identical version, all consumers dep on it unversioned) |

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

## ROCm 10 column — RESULT-PENDING

After hr-fleet reports the merge green: rebuild EVERYTHING in a second fresh dir
(`/var/cache/rivoli/target/rocm10-mig-new`); deviceless suite; flocked battery against the
SAME vendored goldens at the SAME tolerances (a trip means codegen moved numerics past a
measured floor — investigate, never widen); pinned decode ×2 (byte-identity is THE
toolchain-determinism instrument — the baseline column proves it holds under 7.14, so a
divergence here is the toolchain's); GLM A-vs-A at 100 GiB; `tests/parity-glm.sh` full run;
bench A/B against the numbers above.

**Decision rule** (pre-registered): any red that is not a measured-and-accepted improvement
rolls back — one message to hr-fleet, `emerge --usepkgonly`, rivoli rebuilt under 7.14, and
this doc's verdict records the rollback and why.
