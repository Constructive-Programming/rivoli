// io_uring + O_DIRECT into VMM: does NVMe give MORE bandwidth at queue depth?
// (QD=1 ~= the single-thread O_DIRECT 4GB/s; if QD scales, async streaming wins.)
#include <hip/hip_runtime.h>
#include <liburing.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <ctime>
#define CK(x) do{hipError_t e=(x); if(e!=hipSuccess){printf("FATAL %s:%s\n",#x,hipGetErrorString(e));return 1;}}while(0)
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec/1e9;}
static void* vmm(int dev,size_t N){
  hipMemAllocationProp p{};p.type=hipMemAllocationTypePinned;p.location.type=hipMemLocationTypeDevice;p.location.id=dev;
  size_t g=0;hipMemGetAllocationGranularity(&g,&p,hipMemAllocationGranularityMinimum);size_t sz=((N+g-1)/g)*g;
  hipMemGenericAllocationHandle_t h;if(hipMemCreate(&h,sz,&p,0)!=hipSuccess)return nullptr;
  void*ptr=nullptr;hipMemAddressReserve(&ptr,sz,0,nullptr,0);hipMemMap(ptr,sz,0,h,0);
  hipMemAccessDesc d{};d.location.type=hipMemLocationTypeDevice;d.location.id=dev;d.flags=hipMemAccessFlagsProtReadWrite;hipMemSetAccess(ptr,sz,&d,1);
  hipMemAccessDesc hd{};hd.location.type=hipMemLocationTypeHost;hd.location.id=0;hd.flags=hipMemAccessFlagsProtReadWrite;hipMemSetAccess(ptr,sz,&hd,1);
  return ptr;
}
static double bench(int fd,unsigned char*buf,size_t total,size_t chunk,int qd){
  struct io_uring ring; if(io_uring_queue_init(qd,&ring,0)<0)return -1;
  size_t nch=total/chunk, sub=0, comp=0, next=0;
  double t0=now();
  auto push=[&](){ struct io_uring_sqe*s=io_uring_get_sqe(&ring); off_t off=(off_t)next*chunk;
    io_uring_prep_read(s,fd,buf+off,chunk,off); next++; sub++; };
  while(sub<nch && (int)(sub-comp)<qd) push();
  io_uring_submit(&ring);
  while(comp<nch){
    struct io_uring_cqe*c; io_uring_wait_cqe(&ring,&c);
    if(c->res<0){printf("  read err %d\n",c->res);}
    io_uring_cqe_seen(&ring,c); comp++;
    if(sub<nch){ push(); io_uring_submit(&ring); }
  }
  double dt=now()-t0; io_uring_queue_exit(&ring);
  return total/1e6/dt;
}
int main(int argc,char**argv){
  int dev=0; CK(hipSetDevice(dev));
  struct stat st; stat(argv[1],&st);
  size_t chunk=1<<20;
  size_t total=(size_t)2<<30; if(total>(size_t)st.st_size-chunk) total=((st.st_size-chunk)/chunk)*chunk;
  int fd=open(argv[1],O_RDONLY|O_DIRECT); if(fd<0){perror("open");return 1;}
  unsigned char*buf=(unsigned char*)vmm(dev,total); if(!buf){printf("vmm fail\n");return 1;}
  printf("Reading %zu MiB O_DIRECT into VMM, 1MiB chunks:\n", total>>20);
  for(int qd:{1,4,16,32,64,128}){
    double mbps=bench(fd,buf,total,chunk,qd);
    printf("  QD=%-4d %.0f MB/s (%.1f GB/s)\n", qd, mbps, mbps/1000.0);
  }
  // correctness: sample-compare after last run
  int fb=open(argv[1],O_RDONLY); unsigned char*ref=(unsigned char*)malloc(chunk);
  pread(fb,ref,chunk,0); int ok=memcmp(buf,ref,chunk)==0; close(fb);
  printf("correctness (first chunk): %s\n", ok?"MATCH":"MISMATCH");
  return 0;
}
