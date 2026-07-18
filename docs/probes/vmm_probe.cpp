// Feasibility probe: can we get device-local (MTYPE_RW, full-bandwidth) memory
// that the CPU can also fill directly (single copy, no H2D)? Compares three
// allocations head-to-head with the SAME streaming-read kernel:
//   A) hipMalloc            — device-local, NOT host-writable (baseline fast)
//   B) hipHostMalloc coherent — host-fillable, the ~9% tax
//   C) VMM device-local + hipMemSetAccess(Host) — the holy-grail candidate
// Reports host-fillability of C and read BW of all three.
//
// build: hipcc -O3 --offload-arch=gfx1151 vmm_probe.cpp -o vmm_probe
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>

#define CK(x) do{ hipError_t e=(x); if(e!=hipSuccess){ \
  printf("FATAL %s: %s (%d)\n",#x,hipGetErrorString(e),e); return 1; } }while(0)

// Streaming read-once BW kernel: sum every float, block-reduce to out[block] so
// the loads can't be optimized away.
__global__ void bw(const float* __restrict__ p, size_t n, float* out){
  size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x, stride=(size_t)gridDim.x*blockDim.x;
  float acc=0.0f; for(size_t k=i;k<n;k+=stride) acc+=p[k];
  __shared__ float s[256]; s[threadIdx.x]=acc; __syncthreads();
  for(int d=128; d>0; d>>=1){ if(threadIdx.x<d) s[threadIdx.x]+=s[threadIdx.x+d]; __syncthreads(); }
  if(threadIdx.x==0) out[blockIdx.x]=s[0];
}

static double measure(const float* dptr, size_t nf, float* out, int blocks){
  hipLaunchKernelGGL(bw, dim3(blocks), dim3(256), 0, 0, dptr, nf, out); // warmup
  hipDeviceSynchronize();
  hipEvent_t a,b; hipEventCreate(&a); hipEventCreate(&b);
  const int it=30; hipEventRecord(a);
  for(int i=0;i<it;i++) hipLaunchKernelGGL(bw, dim3(blocks), dim3(256), 0, 0, dptr, nf, out);
  hipEventRecord(b); hipEventSynchronize(b);
  float ms=0; hipEventElapsedTime(&ms,a,b); hipEventDestroy(a); hipEventDestroy(b);
  return (double)(nf*sizeof(float))*it / (ms/1e3) / 1e9; // GB/s
}

int main(){
  int dev=0; CK(hipSetDevice(dev));
  const size_t BYTES=(size_t)1<<30;                  // 1 GiB
  const int blocks=2048;
  size_t nf=BYTES/sizeof(float);
  float* out; CK(hipMalloc(&out, (size_t)blocks*sizeof(float)));

  // A) hipMalloc device-local
  { float* p; CK(hipMalloc(&p, BYTES));
    printf("A hipMalloc(device)       read BW: %6.1f GB/s\n", measure(p,nf,out,blocks));
    hipFree(p); }

  // B) hipHostMalloc coherent (host-fillable, the taxed path)
  { float* p; CK(hipHostMalloc((void**)&p, BYTES, 0x40000000/*Coherent*/|0x2/*Mapped*/));
    for(size_t i=0;i<nf;i+=1024) p[i]=1.0f;          // CPU fill works by construction
    printf("B hipHostMalloc(coherent) read BW: %6.1f GB/s  [CPU-fillable: YES]\n", measure(p,nf,out,blocks));
    hipHostFree(p); }

  // C) VMM device-local + try to grant HOST access
  { hipMemAllocationProp prop{};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = dev;
    size_t gran=0; CK(hipMemGetAllocationGranularity(&gran,&prop,hipMemAllocationGranularityMinimum));
    size_t size=((BYTES+gran-1)/gran)*gran;
    hipMemGenericAllocationHandle_t h; CK(hipMemCreate(&h,size,&prop,0));
    void* ptr=nullptr; CK(hipMemAddressReserve(&ptr,size,0,nullptr,0));
    CK(hipMemMap(ptr,size,0,h,0));
    hipMemAccessDesc dd{}; dd.location.type=hipMemLocationTypeDevice; dd.location.id=dev;
    dd.flags=hipMemAccessFlagsProtReadWrite; CK(hipMemSetAccess(ptr,size,&dd,1));

    // THE CRUX: grant CPU access to device-local memory.
    hipMemAccessDesc hd{}; hd.location.type=hipMemLocationTypeHost; hd.location.id=0;
    hd.flags=hipMemAccessFlagsProtReadWrite;
    hipError_t he=hipMemSetAccess(ptr,size,&hd,1);
    bool hostOk=(he==hipSuccess);
    printf("C VMM host-access grant: %s (%d)\n", hipGetErrorString(he), he);

    size_t vnf=size/sizeof(float);
    bool wrote=false;
    if(hostOk){
      // FULL CPU fill with 1.0 → GPU sum must equal vnf (proves CPU→GPU coherence,
      // not just that the write didn't fault).
      float* fp=(float*)ptr; for(size_t i=0;i<vnf;i++) fp[i]=1.0f;
      wrote=true;
    }
    double bw_c = measure((const float*)ptr, vnf, out, blocks);

    // INTERLEAVED: CPU re-writes a chunk, then GPU reads all — the cold-pool
    // pattern (re-filled every miss). If this drops to host speed, the coherence
    // cost is paid per read because the CPU just dirtied the pages; write-once
    // (resident pin) keeps device speed (bw_c above).
    if(hostOk){
      float* fp=(float*)ptr;
      hipEvent_t a,b; hipEventCreate(&a); hipEventCreate(&b);
      const int it=30; hipEventRecord(a);
      for(int i=0;i<it;i++){
        for(size_t k=0;k<vnf;k+=256) fp[k]=(float)i;      // CPU re-dirty (sparse)
        hipLaunchKernelGGL(bw,dim3(blocks),dim3(256),0,0,(const float*)ptr,vnf,out);
      }
      hipEventRecord(b); hipEventSynchronize(b);
      float ms=0; hipEventElapsedTime(&ms,a,b); hipEventDestroy(a); hipEventDestroy(b);
      printf("C VMM interleaved (rewrite+read) BW: %6.1f GB/s  [cold-pool pattern]\n",
             (double)(vnf*sizeof(float))*it/(ms/1e3)/1e9);
    }
    // Coherence check: sum the block partials the kernel just wrote.
    float* hout=(float*)malloc((size_t)blocks*sizeof(float));
    CK(hipMemcpy(hout, out, (size_t)blocks*sizeof(float), hipMemcpyDeviceToHost));
    double gpu_sum=0; for(int i=0;i<blocks;i++) gpu_sum+=hout[i]; free(hout);
    double want=(double)vnf; double err=(gpu_sum-want)/want;
    printf("C VMM(device-local)       read BW: %6.1f GB/s  [CPU-fillable: %s]\n",
           bw_c, hostOk ? (wrote?"YES":"grant-ok-unwritten") : "NO");
    printf("C coherence: GPU sum=%.0f expected=%.0f rel_err=%.2e -> %s\n",
           gpu_sum, want, err, (err>-1e-3 && err<1e-3) ? "CPU->GPU COHERENT" : "MISMATCH!");
    hipMemUnmap(ptr,size); hipMemRelease(h); hipMemAddressFree(ptr,size); }

  printf("\nVERDICT: C is useful only if it reads at ~A's BW AND CPU-fillable:YES.\n");
  return 0;
}
