// Device-local memory the CPU can fill in place — the escape from the coherent
// read tax (see docs/reference/architecture.md §1). On this APU (gfx1151, unified
// LPDDR5X) `hipHostMalloc` of any grain reads ~9% slower than `hipMalloc` because
// it maps system-domain (MTYPE_NC/UC + PTE_SYSTEM + snoop, 4 KiB pages); no
// host-malloc flag crosses into device-local. The HIP VMM API does: allocate
// DEVICE-LOCAL physical memory (MTYPE_RW, full GPU bandwidth) then grant the CPU
// access too, so weights `pread`/memcpy straight in and the GPU reads them in
// place — single copy, no H2D, no tax. Verified: 220 GB/s (= hipMalloc) AND
// CPU->GPU coherent.
//
// SOURCES, corrected 2026-08-01. This cited `docs/hip-apu-memory.md`, deleted in
// the empty-slate rebuild, and `docs/measurement/probes/vmm_probe.cpp`, which was
// never at that path — the probe is `docs/probes/vmm_probe.cpp`, deleted in the
// same commit and recoverable with `git show 3e1bd96:docs/probes/vmm_probe.cpp`.
// So the ~9% read figure has no live measurement behind it; read it as the
// rationale of record, not a standing number. What IS measured today is the write
// side, in docs/reference/architecture.md §3: 5.66 GB/s DMA into VMM pages against
// 12.4 GB/s into the pinned arena.
//
// Host-only HIP (no kernels); compiled into librivolikernels.a like the rest.
// Return codes: 0 ok, POSITIVE arg guard, NEGATIVE -(hipError_t).
#include <hip/hip_runtime.h>

// Allocate `size` bytes of device-local memory mapped for BOTH device and host
// access. Reports the VA in *out_ptr, the physical handle in *out_handle, and the
// granularity-rounded mapped size in *out_mapped (needed verbatim to free).
extern "C" int rivoli_vmm_alloc(size_t size, int dev, void** out_ptr,
                                unsigned long long* out_handle, size_t* out_mapped) {
  if (size == 0) return 1001;

  hipMemAllocationProp prop = {};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = dev;

  size_t gran = 0;
  hipError_t e = hipMemGetAllocationGranularity(&gran, &prop,
                                                hipMemAllocationGranularityMinimum);
  if (e != hipSuccess) return -(int)e;
  size_t msize = ((size + gran - 1) / gran) * gran;

  hipMemGenericAllocationHandle_t h;
  e = hipMemCreate(&h, msize, &prop, 0);
  if (e != hipSuccess) return -(int)e;

  void* ptr = nullptr;
  e = hipMemAddressReserve(&ptr, msize, 0, nullptr, 0);
  if (e != hipSuccess) { hipMemRelease(h); return -(int)e; }

  e = hipMemMap(ptr, msize, 0, h, 0);
  if (e != hipSuccess) { hipMemAddressFree(ptr, msize); hipMemRelease(h); return -(int)e; }

  // Device access, then host access (the domain-crossing grant). Two calls,
  // matching the proven probe sequence.
  hipMemAccessDesc dd = {};
  dd.location.type = hipMemLocationTypeDevice; dd.location.id = dev;
  dd.flags = hipMemAccessFlagsProtReadWrite;
  e = hipMemSetAccess(ptr, msize, &dd, 1);
  if (e == hipSuccess) {
    hipMemAccessDesc hd = {};
    hd.location.type = hipMemLocationTypeHost; hd.location.id = 0;
    hd.flags = hipMemAccessFlagsProtReadWrite;
    e = hipMemSetAccess(ptr, msize, &hd, 1);
  }
  if (e != hipSuccess) {
    hipMemUnmap(ptr, msize); hipMemAddressFree(ptr, msize); hipMemRelease(h);
    return -(int)e;
  }

  *out_ptr = ptr;
  *out_handle = (unsigned long long)h;
  *out_mapped = msize;
  return 0;
}

// Tear down a rivoli_vmm_alloc mapping. `mapped` MUST be the value it returned.
extern "C" int rivoli_vmm_free(void* ptr, unsigned long long handle, size_t mapped) {
  hipError_t e1 = hipMemUnmap(ptr, mapped);
  hipError_t e2 = hipMemRelease((hipMemGenericAllocationHandle_t)handle);
  hipError_t e3 = hipMemAddressFree(ptr, mapped);
  if (e1 != hipSuccess) return -(int)e1;
  if (e2 != hipSuccess) return -(int)e2;
  if (e3 != hipSuccess) return -(int)e3;
  return 0;
}

// Fill `bytes` at `dst` with the 32-bit pattern `pat`. Used to POISON a slot the moment
// it is admitted but before its bytes land.
//
// The point is to make a read-before-write DETERMINISTIC. Today an unloaded slot holds
// whatever was there: uninitialised memory (-> NaN, ~6% of long runs) or a previous
// expert's weights (-> finite, plausible, and silently WRONG). Poisoning with a quiet-NaN
// pattern collapses both into the loud case, so a fault that reproduces one run in
// sixteen reproduces every time it actually occurs — and the silent-corruption mode
// becomes observable at all, which it currently is not.
extern "C" __global__ void fill_u32(unsigned* __restrict__ dst, unsigned pat, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = pat;
}

extern "C" int rivoli_fill_u32(void* dst, unsigned pat, size_t bytes) {
    if (bytes % 4 != 0) return 1001;
    size_t n = bytes / 4;
    if (n == 0) return 0;
    hipGetLastError();
    size_t blocks = (n + 255) / 256;
    hipLaunchKernelGGL(fill_u32, dim3(blocks), dim3(256), 0, 0, (unsigned*)dst, pat, n);
    hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -(int)e;
}
