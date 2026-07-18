// Device-local memory the CPU can fill in place — the escape from the coherent
// read tax (see docs/hip-apu-memory.md). On this APU (gfx1151, unified LPDDR5X)
// `hipHostMalloc` of any grain reads ~9% slower than `hipMalloc` because it maps
// system-domain (MTYPE_NC/UC + PTE_SYSTEM + snoop, 4 KiB pages); no host-malloc
// flag crosses into device-local. The HIP VMM API does: allocate DEVICE-LOCAL
// physical memory (MTYPE_RW, full GPU bandwidth) then grant the CPU access too, so
// weights `pread`/memcpy straight in and the GPU reads them in place — single
// copy, no H2D, no tax. Verified: 220 GB/s (= hipMalloc) AND CPU->GPU coherent
// (scratchpad/vmm_probe.cpp).
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
