# HIP memory on Strix Halo (gfx1151) — the coherent-pin tax, traced to the metal

Reference notes for anyone touching the resident-pin / cold-feed memory model.
Everything here is **source- and measurement-grounded**, not inferred. Dates: the
investigation ran 2026-07-18 on rh-anine (AMD RYZEN AI MAX+ 395 / Radeon 8060S,
128 GB LPDDR5X unified).

## The one-line finding

On this APU there is a **~9 % GPU read-bandwidth tax on ANY `hipHostMalloc`
allocation vs `hipMalloc`**, and **no `hipHostMalloc` flag removes it** — because
the tax is *system-memory domain vs device-local domain*, not coherence grain,
and the coherence flag cannot move an allocation across that boundary.

Measured (32-tok GLM-5.2 decode, cold-expert pool = the variable, `mlp` bucket =
GPU reading experts during the MoE dot):

| cold-pool memory | `mlp` ms/tok | `fetch` ms/tok |
|---|---|---|
| `hipMalloc` (device-local) — baseline | **616** | 447 (1.33/miss) |
| `hipHostMalloc` Coherent (fine-grained) | 672 | 368 (1.09/miss) |
| `hipHostMalloc` NonCoherent (coarse) | 678 | 364 (1.08/miss) |
| Coherent + `madvise(MADV_HUGEPAGE)` | 673 | 369 |
| NonCoherent + hugepage | 676 | 364 |

Every host-pool variant is 672–678 regardless of grain; only device-local hits
616. **Grain is not the axis. Domain is.**

## Why — the full chain (kernel + userspace, all readable)

**Kernel (`/usr/src/linux`, on-box — Gentoo ships the source):**
- `amdkfd/kfd_svm.c::svm_range_get_pte_flags` and
  `amdgpu/amdgpu_amdkfd_gpuvm.c::get_pte_flags` → `amdgpu/gmc_v11_0.c::
  gmc_v11_0_get_vm_pte`. gfx1151 = GC IP 11.5.1, which hits the **`default`** arm:
  `coherent ? MTYPE_UC : MTYPE_NC`. The graphics tail also forces `MTYPE_UC` if the
  BO carries `AMDGPU_GEM_CREATE_COHERENT/EXT_COHERENT/UNCACHED`.
- So fine-grained coherent host mem → **`MTYPE_UC` (uncached, L2 bypassed)**;
  coarse/default host mem → **`MTYPE_NC` (L2-cacheable)**. BUT both are system
  domain: PTE carries `AMDGPU_PTE_SYSTEM | AMDGPU_PTE_SNOOPED`, 4 KiB pages.
- Device-local (`hipMalloc`, GPU-agent pool) → `MTYPE_RW`, GPU-owned, large pages,
  **no snoop** → the fast path.

**Userspace (`ROCm/clr`, GitHub `amd-staging` — NOT on-box, only the `.so`):**
- `hipamd/src/hip_memory.cpp::ihipHostMalloc`: `NonCoherent` clears
  `CL_MEM_SVM_ATOMICS`.
- `rocclr/device/rocm/rocmemory.hpp::getHostMemorySegment`: `ATOMICS==0` →
  `kNoAtomics`.
- `rocclr/device/rocm/rocdevice.cpp::getHostMemoryPool`: `kNoAtomics` →
  `coarse_grain_pool` **iff its handle ≠ 0**, else falls through to the fine
  (coherent) pool.
- `rocminfo`: the **CPU agent DOES expose a `COARSE GRAINED` ~128 GB pool**. So
  `NonCoherent` genuinely reaches the coarse/`MTYPE_NC`/cached pool — and it still
  measured 678, because for a **streaming read-once** of 18.9 MB experts (no L2
  reuse) `NC`-cacheable vs `UC`-uncached barely matters; the residual gap is
  `PTE_SYSTEM`+snoop + 4 KiB-page TLB, i.e. the domain, which the flag can't cross.

`madvise(MADV_HUGEPAGE)` was a no-op (HSA pool memory isn't THP-eligible), so the
TLB component wasn't reachable that way either.

## Consequence for the coherent resident pin ("A3")

A3's premise — one host copy, `pread` straight in, GPU reads in place — is
**inherently a system-domain read**, so it pays the ~9 % intrinsically. You cannot
get single-copy AND device bandwidth from `hipHostMalloc`. The only zero-tax
memory is device-local (`hipMalloc`) = the double-store A3 wanted to remove.

The `fetch`-fill offset (447→~365 ms, host memcpy skips 3× H2D/miss) is real and
lever-independent — the one genuine upside of a host-fillable pool.

## Escape hatches (different mechanisms, uncertain payoff)

1. **Large-page system mapping** via `hipHostRegister` over a `MAP_HUGETLB`
   region — attacks only the TLB portion of the tax (needs reserved hugepages;
   `/proc/sys/vm/nr_hugepages` is 0 by default here).
2. **VMM / `hipMemPool`** — allocate device-local physical memory
   (`hipMemCreate` with `hipMemLocationTypeDevice`) and try to grant the CPU
   access (`hipMemSetAccess` with `hipMemLocationTypeHost = 2`). If the APU allows
   host access to a device-local VMM allocation, that's single-copy AND
   device-bandwidth — the holy grail. **See the probe result below.**

## VMM device-local host-fill — IT WORKS (with one caveat)

The escape hatch is real. Allocate device-local physical memory
(`hipMemCreate`, `hipMemLocationTypeDevice`), map it, and grant the CPU access
too (`hipMemSetAccess` with a **second** desc at `hipMemLocationTypeHost`). On this
APU the host grant succeeds, the CPU writes in place, and the GPU reads it at
**device bandwidth** — verified CPU→GPU coherent (probe: GPU sum of a full CPU
fill = exact, rel_err 0). This is the primitive `device::VmmBuf` (via the C ABI
shim `kernels/vmm.hip`: `rivoli_vmm_alloc`/`rivoli_vmm_free`).

Microbench (`docs/probes/vmm_probe.cpp`, 1 GiB, gfx1151):

| memory | read BW | CPU-fillable |
|---|---|---|
| `hipMalloc` (device) | 221 GB/s | no |
| `hipHostMalloc` coherent | ~215 GB/s | yes |
| **VMM device-local + host grant** | **220 GB/s** | **yes** |
| VMM, interleaved rewrite+read | **112 GB/s** | yes |

**THE CAVEAT (measured, decisive):** VMM gives device bandwidth only for
**write-once-read-many**. If the CPU re-dirties the pages before each GPU read,
coherence traffic halves the read (220→112 GB/s). So:
- **Cold-expert pool** (re-filled every miss → CPU-dirty-then-read): VMM does NOT
  beat the read tax — real-decode `mlp` stayed 673 (like `hipHostMalloc`), because
  each slot is CPU-written then GPU-read once. It IS still a small net win there
  via the fill offset (host memcpy 364 vs H2D 447 ms `fetch`) → wall 2086 vs 2136.
  That's why the cold pool now uses `VmmBuf` — fill-dominated net win, not a read
  win.
- **Resident pin (A3)**: filled ONCE at build, read every token for the whole run
  = the write-once-read-many pattern = the 220 GB/s device path. **This is where
  VMM pays off**: single host copy (drop the mmap page-cache duplicate AND the
  device-tier `hipMalloc` slab — no double-store), device bandwidth (no ~9% tax),
  and ~2× experts fit the same RAM. The microbench + interleave test de-risk it;
  the resident-tier VMM conversion is the concrete A3 build.

Net: my "you can't have single-copy AND device bandwidth from host memory" was
true for `hipHostMalloc` but FALSE in general — VMM crosses the domain boundary.
The constraint that remains is the access *pattern* (write-once), not the API.

## Epistemic note

The first pass asserted "no flag fixes it, fundamental" from black-box behavior
alone — right conclusion, wrong stated mechanism (blamed coherent→uncached).
Reading the kernel MTYPE logic in isolation then over-corrected to "avoidable".
Only the full triangulation — kernel PTE source + clr pool trace + `rocminfo` +
our own per-variant data — settled it correctly. **The source was on the box the
whole time; read it before saying "definitively".**
