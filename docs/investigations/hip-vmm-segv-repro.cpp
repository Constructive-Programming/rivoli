// Minimal reproducer: SIGSEGV inside libamdhip64's VMM path on gfx1151 (Strix Halo APU).
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
//
// Observed with HIP 7.2.53210. The fault is a NULL dereference inside the runtime:
//
//   kernel: rivoli-vmm[2141025]: segfault at e0 ip 00007fbaa865b66b error 4 in
//           libamdhip64.so.7.2.53210[45b66b,...]
//   code:   48 8b 90 e0 00 00 00   ->  mov rdx,[rax+0xe0]   with rax = 0
//
// and it lands inside hipMemCreate/hipMemAddressReserve/hipMemMap (2 of 3 captured cores) or
// inside hipMemUnmap/hipMemRelease/hipMemAddressFree (1 of 3).
//
// WHAT MATTERS, from bisecting a Rust application down to this:
//   * ~1500 cycles of pure VMM alloc/free on ONE thread: clean.
//   * The same VMM churn PLUS ordinary HIP work (hipMalloc, a kernel, a sync) on ONE thread:
//     clean.
//   * The same work split across TWO threads that never run concurrently: crashes.
// So it is not concurrency and not churn on its own — it is that a second thread participates.
//
// Build:  hipcc -O2 -o vmm_repro vmm_repro.cpp
// Run:    ./vmm_repro [cycles_per_thread] [threads] [bytes]
// Exit:   0 = completed, 1 = a HIP call returned an error, or SIGSEGV.
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#define CHECK(expr)                                                                    \
  do {                                                                                 \
    hipError_t _e = (expr);                                                            \
    if (_e != hipSuccess) {                                                             \
      std::fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #expr,             \
                   hipGetErrorString(_e));                                             \
      std::exit(1);                                                                     \
    }                                                                                  \
  } while (0)

// How much page-cache-backed memory each cycle copies into the VMM mapping. The mapping is
// created at exactly this size, so the copy below cannot run off the end — the first version of
// this file read 64 KB from a 7 KB file and died on SIGBUS, which is not the SIGSEGV under test.
static constexpr size_t SRC_BYTES = 65536;

static void* g_map = nullptr;

__global__ void touch(float* p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = p[i] + 1.0f;
}

// The application's allocator, transliterated: device-local physical memory mapped for BOTH
// device and host access, so the CPU can fill weights in place with no H2D copy.
static void vmm_alloc(size_t size, int dev, void** out_ptr, hipMemGenericAllocationHandle_t* out_h,
                      size_t* out_mapped) {
  hipMemAllocationProp prop = {};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = dev;

  size_t gran = 0;
  CHECK(hipMemGetAllocationGranularity(&gran, &prop, hipMemAllocationGranularityMinimum));
  size_t msize = ((size + gran - 1) / gran) * gran;

  hipMemGenericAllocationHandle_t h;
  CHECK(hipMemCreate(&h, msize, &prop, 0));

  void* ptr = nullptr;
  CHECK(hipMemAddressReserve(&ptr, msize, 0, nullptr, 0));
  CHECK(hipMemMap(ptr, msize, 0, h, 0));

  hipMemAccessDesc dd = {};
  dd.location.type = hipMemLocationTypeDevice;
  dd.location.id = dev;
  dd.flags = hipMemAccessFlagsProtReadWrite;
  CHECK(hipMemSetAccess(ptr, msize, &dd, 1));

  hipMemAccessDesc hd = {};
  hd.location.type = hipMemLocationTypeHost;
  hd.location.id = 0;
  hd.flags = hipMemAccessFlagsProtReadWrite;
  CHECK(hipMemSetAccess(ptr, msize, &hd, 1));

  *out_ptr = ptr;
  *out_h = h;
  *out_mapped = msize;
}

static void vmm_free(void* ptr, hipMemGenericAllocationHandle_t h, size_t mapped) {
  CHECK(hipMemUnmap(ptr, mapped));
  CHECK(hipMemRelease(h));
  CHECK(hipMemAddressFree(ptr, mapped));
}

static void cycle(int id, int cycles, size_t bytes) {
  for (int i = 0; i < cycles; ++i) {
    void* vp = nullptr;
    hipMemGenericAllocationHandle_t h;
    size_t mapped = 0;

    // Sizes vary per cycle, as the application's do (a differently-sized tier per engine):
    // every distinct size is a distinct granularity rounding and a distinct VA reservation.
    vmm_alloc(bytes + (size_t)(i % 5) * 4096, 0, &vp, &h, &mapped);

    // The CPU fills the mapping in place — the reason for using VMM at all.
    std::memset(vp, 0, 4096);

    // The application fills the VMM slab by memcpy FROM AN MMAPPED FILE (its weight artifact),
    // so the source pages are page-cache, not anonymous memory. `main` refuses to start without
    // the mapping, so this is unconditional and cannot be skipped into a weaker run.
    std::memcpy(vp, g_map, SRC_BYTES);

    // ~20 plain device allocations per cycle, which is what one engine makes: a K and a V
    // buffer per layer plus activations. One hipMalloc per cycle did NOT reproduce.
    float* d[24];
    for (int k = 0; k < 24; ++k) CHECK(hipMalloc(&d[k], 4096));
    hipLaunchKernelGGL(touch, dim3(1), dim3(64), 0, 0, d[0], 64);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    for (int k = 0; k < 24; ++k) CHECK(hipFree(d[k]));

    vmm_free(vp, h, mapped);

    if ((i + 1) % 20 == 0) std::fprintf(stderr, "  thread %d: %d cycles\n", id, i + 1);
  }
}

int main(int argc, char** argv) {
  int cycles = argc > 1 ? std::atoi(argv[1]) : 40;
  int nthreads = argc > 2 ? std::atoi(argv[2]) : 2;
  size_t bytes = argc > 3 ? (size_t)std::atoll(argv[3]) : 1425856;
  if (bytes < SRC_BYTES) {
    std::fprintf(stderr, "bytes must be >= %zu, the size of the memcpy source\n", SRC_BYTES);
    return 1;
  }

  int dev_count = 0;
  CHECK(hipGetDeviceCount(&dev_count));
  hipDeviceProp_t p;
  CHECK(hipGetDeviceProperties(&p, 0));
  std::fprintf(stderr, "device 0: %s (%s), HIP %d.%d\n", p.name, p.gcnArchName, HIP_VERSION_MAJOR,
               HIP_VERSION_MINOR);
  std::fprintf(stderr, "%d thread(s) x %d cycles, base %zu bytes\n", nthreads, cycles, bytes);
  // A file to memcpy from, standing in for the weight artifact's mmap — CREATED here rather than
  // assumed. An earlier version opened a path this program does not write and fell through to
  // `g_map = nullptr` on failure, so the `memcpy` below was skipped and the run still exited 0
  // while claiming to have exercised a page-cache source. That is the same class as the SIGBUS
  // this file already records: a silently weakened reproducer reads as a negative result.
  {
    char path[] = "/tmp/vmm_repro_src_XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) {
      std::perror("mkstemp");
      return 1;
    }
    unlink(path); // the fd keeps it alive; nothing is left behind
    std::vector<char> filler(SRC_BYTES, 0x5a);
    if (write(fd, filler.data(), filler.size()) != (ssize_t)filler.size()) {
      std::perror("write");
      return 1;
    }
    g_map = mmap(nullptr, SRC_BYTES, PROT_READ, MAP_PRIVATE, fd, 0);
    if (g_map == MAP_FAILED) {
      std::perror("mmap");
      return 1;
    }
    close(fd); // the mapping survives the descriptor
  }

  // SEQUENTIAL threads: each is joined before the next starts, so no two ever run at the same
  // time. This is what libtest does with --test-threads=1, and it is the shape that fails.
  for (int t = 0; t < nthreads; ++t) {
    std::thread th(cycle, t, cycles, bytes);
    th.join();
  }
  std::fprintf(stderr, "completed\n");
  return 0;
}
