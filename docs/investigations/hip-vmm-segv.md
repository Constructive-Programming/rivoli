---
scope: engine
status: live
verdict: A SIGSEGV inside libamdhip64's virtual-memory path on gfx1151/HIP 7.2.53210, hit by tests/glimmer_reference.rs at ~6% of runs and by a two-thread synthetic at 25%. The fault is a NULL dereference in the runtime -- `mov rdx,[rax+0xe0]` with rax=0, `segfault at e0 error 4` -- landing in hipMemCreate/AddressReserve/Map (2 of 3 captured cores) or in hipMemUnmap/Release/AddressFree (1 of 3). NOT concurrency: with --test-threads=1 the threads never overlap, and it is the mere PARTICIPATION of a second thread that matters. Bisected: 1500 bare VMM alloc/free cycles on one thread clean, ~1000 engine build/decode/drop cycles in ONE test clean, the same work split across TWO tests 2-in-8 then 3-in-12, all of it marshalled onto one worker thread 0-in-12. Two rivoli-side fixes were tried and MEASURED: hipSetDevice per thread changed nothing (3 in 12), and a dedicated VMM thread fixed the synthetic (25% -> 0 in 15) but NOT glimmer_reference (1 in 15 against 3 in 52 before, i.e. unchanged) -- and the surviving crash is ON the VMM thread, which refutes thread-affinity as the whole cause. A standalone C reproducer is in hip-vmm-segv-repro.cpp; it does NOT yet fire, which rules out the simple shapes (VMM churn, hipMalloc volume, kernels, syncs, varying sizes, mmap-sourced memcpy, two sequential threads) and is itself part of the report. SUPERSEDED THE SAME DAY: a full ROCm 7.14.0 upgrade landed mid-session at 20:02 and libamdhip64.so.7.2.53210 is DELETED from disk, so every measurement here is against a runtime that no longer exists and none of it has been re-tested. It could not be: pre-upgrade binaries now fail with hipMemGetInfo 100 (hipErrorNoDevice) although the GPU is healthy -- their gfx1151 code objects are what 7.14 rejects -- and hipcc cannot rebuild them because clang-23.1.0_rc3 dies on an undefined symbol in its own libclang-cpp. The next action is to make the toolchain consistent, rebuild, and re-run tests/vmm_threads.rs and glimmer_reference in a loop.
---

# A SIGSEGV inside HIP's VMM path — gfx1151, HIP 7.2.53210

> **THE RUNTIME THIS WAS MEASURED ON NO LONGER EXISTS ON THIS BOX (2026-08-14, 20:02).** A full
> ROCm **7.14.0** upgrade landed mid-session — `dev-util/hip-7.14.0`, plus `rocm-core`,
> `rocm-comgr` and `rocm-device-libs` — and `libamdhip64.so.7` now resolves to
> **7.14.60850-0000000**. The 7.2.53210 object is deleted, not shadowed.
>
> **Everything below is evidence against 7.2.53210 and has NOT been re-tested on 7.14.0.** That
> is the first question any bug report gets, so it goes at the top rather than in a footnote.
>
> **It could not be re-tested, for two reasons that are worth writing down:**
>
> * **Binaries built before the upgrade fail with `hipMemGetInfo failed (100)`** —
>   `hipErrorNoDevice` — although the GPU is healthy (`rocminfo` sees gfx1151, `rocm-smi` reports
>   it, `/dev/kfd` and both topology nodes are present). The soname did not change, so old
>   binaries load 7.14 and their gfx1151 code objects, built by the 7.2 toolchain, are what the
>   runtime rejects. Everything needs rebuilding.
> * **`hipcc` cannot compile.** `clang++` from LLVM 23 dies on an undefined symbol in its own
>   `libclang-cpp.so.23.1` (`LLVM_23.1`); the installed compiler is `clang-23.1.0_rc3`, built the
>   same evening. A two-line kernel does not build, so the rebuild above is blocked until the
>   toolchain is consistent — `clang-22.1.8` is still installed and is what built this tree
>   earlier the same day.
>
> **So the next action is unchanged in shape and cheaper than it was**: rebuild on 7.14.0 and
> re-run `cargo test --features rocm --test vmm_threads` and `--test glimmer_reference` in a
> loop. If the crash is gone, this file becomes a note. If it survives a jump from 7.2.0 to
> mainline, that is the strongest line the report can carry.

**Not root-caused, and the point of this file is that the evidence is worth more than the
guesses.** Two hypotheses have already been killed by their own measurements.

## The fault

```
kernel: rivoli-vmm[2141025]: segfault at e0 ip 00007fbaa865b66b error 4 in
        libamdhip64.so.7.2.53210[45b66b,7fbaa8212000+487000]
code:   48 8b 90 e0 00 00 00   ->   mov rdx,[rax+0xe0]      with rax = 0
```

A null internal object, dereferenced at a fixed offset. `error 4` is a user-mode read of a
non-present page. **Read the cores with `coredumpctl debug <pid> --debugger=gdb`** — this box
runs `ptrace_scope=1`, which blocks a live attach but not core analysis, and that distinction
cost an hour the first time.

| core | rivoli frame | under |
|---|---|---|
| 2041319 | `rivoli_vmm_free` | `Drop for VmmBuf` → `DeviceTier` → `GlimmerPin` → `Glimmer` |
| 2019129 | `rivoli_vmm_alloc` | `VmmBuf::new` → `DeviceTier::new` → `GlimmerPin::build` |
| 2091304 | `rivoli_vmm_alloc` | same, via `common::decode_one` |
| 2141019 | `rivoli_vmm_alloc` | same, but **on the dedicated VMM thread** (see below) |

## The bisect

Every row is `--test-threads=1`, so no two threads ever run at the same time.

| shape | crashes |
|---|---|
| 1500 bare `DeviceTier` alloc/free cycles (`examples/vmm_churn`) | 0 |
| ~1000 engine build/decode/drop cycles in ONE `#[test]` | 0 |
| the same work split across TWO `#[test]`s in one binary | **2 in 8, then 3 in 12** |
| both marshalled onto one worker thread | **0 in 12** |

**So it is not churn and not concurrency — it is that a second thread participates at all.**
libtest gives every `#[test]` its own thread, which is why a test binary finds this and the
single-threaded production decode path does not.

## What was tried, and what the measurements said

**`hipSetDevice(dev)` at the top of the allocator — REFUTED.** Nothing in this tree had ever
called it, and HIP's current device is per-thread state, so this looked obvious. 3 crashes in 12,
against 2 in 8 before. No effect.

**A dedicated VMM thread — HALF a fix, and the surviving half refutes the hypothesis.** Every
`rivoli_vmm_alloc`/`rivoli_vmm_free` now runs on one long-lived thread (`memory::device`'s
`vmm_thread`). The synthetic two-thread gate went **25% → 0 in 15**, which at that rate is
p ≈ 0.013 — a real effect. But `glimmer_reference` went to **1 in 15 against 3 in 52 before**,
i.e. unchanged, **and the surviving crash is on the VMM thread itself**. Serializing the mapping
calls is therefore not sufficient, and thread affinity is at best one of two causes.

## The C reproducer, and why a non-firing one is still evidence

`hip-vmm-segv-repro.cpp` transliterates the allocator and the surrounding work, and **does not
reproduce** across 12 runs of 2 sequential threads × 40 cycles. That rules out, in plain C with
no application in the frame: VMM alloc/free churn, per-cycle `hipMalloc` volume (24 of them),
kernel launches, `hipDeviceSynchronize`, varying allocation sizes, a page-cache `mmap` as the
memcpy source, and two sequential threads. Whatever the remaining ingredient is, it is none of
those.

> One caught mistake worth keeping: v2 of the reproducer failed 12/12 with **SIGBUS**, not
> SIGSEGV — it memcpy'd 64 KB out of a 7 KB file. A gate going red for the wrong reason looks
> exactly like success if the only thing checked is the exit code.

## Do this first: we WERE behind — and the upgrade has since happened

| | version |
|---|---|
| installed here | **HIP 7.2.53210** (`dev-util/hip-7.2.0-r1`) |
| HIP docs "latest" | 7.2.53211 |
| stable in the 7.2 line | **ROCm 7.2.4** (2026-05-29) |
| mainline | **ROCm 7.14.0** |

**Superseded the same day: 7.14.0 is now installed** (see the banner at the top). The table above
is kept as the state the investigation ran in, because every measurement in this file was taken
under it. The available versions in this tree are 6.3.3-r2, 6.4.3-r2, 7.0.2-r1, 7.1.0-r1,
7.2.0-r1 and 7.14.0 — so 7.2.53210 is reinstallable if a bisect between the two ever becomes
worth the time.
