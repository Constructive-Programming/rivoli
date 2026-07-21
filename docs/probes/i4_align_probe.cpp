// Does the +8 byte skew (every snapshot tensor sits at offset ≡ 8 mod 16) cost
// bandwidth on the uint (4 B/lane) path? A wave's 128 B span then straddles two
// 128 B cache lines instead of landing on one.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#define WAVE 32
#define ROWS_PER_BLOCK 8
__device__ __forceinline__ float wave_sum(float v){
    for(int o=WAVE/2;o>0;o>>=1) v+=__shfl_down(v,o,WAVE); return v; }

__global__ void gu_u32(const float* __restrict__ x,int hidden,int inter,int E,
        const unsigned char* __restrict__ gate,const unsigned char* __restrict__ up,
        const float* __restrict__ gs,const float* __restrict__ us,float* __restrict__ h){
    int lane=threadIdx.x&(WAVE-1);
    size_t r=(size_t)blockIdx.x*ROWS_PER_BLOCK+threadIdx.x/WAVE;
    if(r>=(size_t)E*inter) return;
    size_t rb=(size_t)hidden/2;
    const unsigned char* rows[2]={gate+r*rb,up+r*rb};
    float o[2];
    for(int t=0;t<2;++t){
        float acc=0.0f; const unsigned int* rw=(const unsigned int*)rows[t];
        for(int base=0;base<hidden;base+=WAVE*8){
            int col=base+lane*8; unsigned int w=rw[col>>3]; const float* vv=x+col;
#pragma unroll
            for(int k=0;k<8;++k) acc+=vv[k]*(float)((int)((w>>(4*k))&0xFu)-8);
        }
        o[t]=wave_sum(acc);
    }
    if(lane==0) h[r]=o[0]*gs[r]*o[1]*us[r];
}
#define CK(e) do{hipError_t _e=(e); if(_e!=hipSuccess){printf("HIP %s L%d\n",hipGetErrorString(_e),__LINE__);exit(1);} }while(0)
int main(){
    const int hidden=6144,inter=2048,E=9; const size_t rb=hidden/2,n=(size_t)E*inter;
    const size_t bytes=n*rb;
    unsigned char *buf; float *x,*gs,*us,*h;
    CK(hipMalloc(&buf,2*bytes+64)); CK(hipMalloc(&x,hidden*4));
    CK(hipMalloc(&gs,n*4)); CK(hipMalloc(&us,n*4)); CK(hipMalloc(&h,n*4));
    CK(hipMemset(buf,0x53,2*bytes+64)); CK(hipMemset(x,0x3c,hidden*4));
    CK(hipMemset(gs,0x3c,n*4)); CK(hipMemset(us,0x3c,n*4));
    dim3 g((n+ROWS_PER_BLOCK-1)/ROWS_PER_BLOCK),b(ROWS_PER_BLOCK*WAVE);
    hipEvent_t t0,t1; CK(hipEventCreate(&t0)); CK(hipEventCreate(&t1));
    printf("uint (4B/lane) path, gate+up over %.1f MB:\n", 2.0*bytes/1e6);
    int skews[3]={0,8,4};
    for(int si=0;si<3;++si){
        int s=skews[si];
        unsigned char* gp=buf+s; unsigned char* upp=buf+bytes+s;
        for(int w=0;w<3;++w) gu_u32<<<g,b>>>(x,hidden,inter,E,gp,upp,gs,us,h);
        CK(hipDeviceSynchronize()); CK(hipEventRecord(t0));
        for(int r=0;r<30;++r) gu_u32<<<g,b>>>(x,hidden,inter,E,gp,upp,gs,us,h);
        CK(hipEventRecord(t1)); CK(hipEventSynchronize(t1));
        float ms; CK(hipEventElapsedTime(&ms,t0,t1)); double per=ms/30;
        printf("  base %%16 = %2d   %7.3f ms  %6.1f GB/s\n", s, per, 2.0*bytes/(per*1e-3)/1e9);
    }
    return 0;
}
