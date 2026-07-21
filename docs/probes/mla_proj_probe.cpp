// Probe: mla.hip's two kv_b projections (mla_absorb, mla_value) are the only int4
// kernels that never got the D3 wave-per-row rewrite — they are still one thread
// per output element, each thread walking a whole row alone.
//
// mla_value is the pathological one: thread (head,j) reads row j of head `head`,
// so 32 lanes of a wave read 32 DIFFERENT rows 256 B apart — every load
// instruction touches 32 separate cache lines and fetches 32x128 B to use 32x1 B.
// mla_absorb strides better (adjacent threads share a byte) but still only 16 B
// of useful data per wave-load.
//
// Compares shipped vs wave-per-row vs wave-per-row + uint4 packed loads, on the
// real GLM-5.2 MLA shapes (H=64, kv_lora=512, qk_nope=128, v_head=128).
//
// build: hipcc -O3 --offload-arch=gfx1151 mla_proj_probe.cpp -o mla_proj_probe

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#define WAVE 32
#define ROWS_PER_BLOCK 8

__device__ __forceinline__ float wave_sum(float v) {
    for (int o = WAVE / 2; o > 0; o >>= 1) v += __shfl_down(v, o, WAVE);
    return v;
}
__device__ __forceinline__ int nib_i4(const unsigned char* row, int i) {
    unsigned char b = row[i >> 1];
    return ((i & 1) ? (b >> 4) : (b & 0x0F)) - 8;
}

// ================= mla_value =================
// A) shipped: one thread per (head,j), walks the whole kvl row alone.
__global__ void value_shipped(const float* __restrict__ clat,
                              const unsigned char* __restrict__ kvb, const float* __restrict__ sc,
                              int H, int nope, int vh, int kvl, float* __restrict__ ctx) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)H * vh) return;
    int head = idx / vh, j = idx % vh;
    size_t rb = (size_t)(kvl + 1) / 2;
    size_t row_idx = (size_t)head * (nope + vh) + nope + j;
    const unsigned char* row = kvb + row_idx * rb;
    const float* cl = clat + (size_t)head * kvl;
    float acc = 0.0f;
    for (int i = 0; i < kvl; ++i) acc += cl[i] * (float)nib_i4(row, i);
    ctx[(size_t)head * vh + j] = acc * sc[row_idx];
}

// B) wave-per-row (the D3 shape used everywhere else): byte loads, coalesced.
__global__ void value_wave(const float* __restrict__ clat,
                           const unsigned char* __restrict__ kvb, const float* __restrict__ sc,
                           int H, int nope, int vh, int kvl, float* __restrict__ ctx) {
    int lane = threadIdx.x & (WAVE - 1);
    size_t r = (size_t)blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / WAVE;
    if (r >= (size_t)H * vh) return;
    int head = (int)(r / vh), j = (int)(r % vh);
    size_t rb = (size_t)(kvl + 1) / 2;
    size_t row_idx = (size_t)head * (nope + vh) + nope + j;
    const unsigned char* row = kvb + row_idx * rb;
    const float* cl = clat + (size_t)head * kvl;
    float acc = 0.0f;
    for (int i = lane; i < kvl; i += WAVE) acc += cl[i] * (float)nib_i4(row, i);
    acc = wave_sum(acc);
    if (lane == 0) ctx[(size_t)head * vh + j] = acc * sc[row_idx];
}

// C) wave-per-row + uint4 packed load (16 B/lane, 512 B/wave). kvl=512 = 32*16,
// so one uint4 step per lane covers the whole row exactly... use the general loop.
__global__ void value_wave_v4(const float* __restrict__ clat,
                              const unsigned char* __restrict__ kvb, const float* __restrict__ sc,
                              int H, int nope, int vh, int kvl, float* __restrict__ ctx) {
    int lane = threadIdx.x & (WAVE - 1);
    size_t r = (size_t)blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / WAVE;
    if (r >= (size_t)H * vh) return;
    int head = (int)(r / vh), j = (int)(r % vh);
    size_t rb = (size_t)(kvl + 1) / 2;
    size_t row_idx = (size_t)head * (nope + vh) + nope + j;
    const unsigned char* row = kvb + row_idx * rb;
    const float* cl = clat + (size_t)head * kvl;
    float acc = 0.0f;
    for (int base = 0; base < kvl; base += WAVE * 32) {
        int col = base + lane * 32;
        if (col >= kvl) break;
        uint4 w = *(const uint4*)(row + (col >> 1));
        const float* vv = cl + col;
        unsigned int ws[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
        for (int q = 0; q < 4; ++q)
#pragma unroll
            for (int k = 0; k < 8; ++k)
                acc += vv[q * 8 + k] * (float)((int)((ws[q] >> (4 * k)) & 0xFu) - 8);
    }
    acc = wave_sum(acc);
    if (lane == 0) ctx[(size_t)head * vh + j] = acc * sc[row_idx];
}

// ================= mla_absorb =================
// A) shipped: one thread per (head,i); loops the nope rows, strided byte reads.
__global__ void absorb_shipped(const float* __restrict__ q,
                               const unsigned char* __restrict__ kvb, const float* __restrict__ sc,
                               int H, int qh, int nope, int vh, int kvl, float* __restrict__ qabs) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)H * kvl) return;
    int head = idx / kvl, i = idx % kvl;
    size_t rb = (size_t)(kvl + 1) / 2;
    size_t rbase = (size_t)head * (nope + vh);
    const float* qp = q + (size_t)head * qh;
    float acc = 0.0f;
    for (int d = 0; d < nope; ++d)
        acc += qp[d] * (float)nib_i4(kvb + (rbase + d) * rb, i) * sc[rbase + d];
    qabs[(size_t)head * kvl + i] = acc;
}

// B) one block per head, staging the nope rows column-blocked: each thread owns
// one output column i and accumulates over d, but the row bytes are read as uint
// (4 B/lane, 8 columns per lane) so a wave-load covers 128 B instead of 16 B.
__global__ void absorb_vec(const float* __restrict__ q,
                           const unsigned char* __restrict__ kvb, const float* __restrict__ sc,
                           int H, int qh, int nope, int vh, int kvl, float* __restrict__ qabs) {
    int head = blockIdx.x;
    if (head >= H) return;
    size_t rb = (size_t)(kvl + 1) / 2;
    size_t rbase = (size_t)head * (nope + vh);
    const float* qp = q + (size_t)head * qh;
    // Each thread owns 8 consecutive columns; blockDim.x*8 columns per block pass.
    for (int col0 = threadIdx.x * 8; col0 < kvl; col0 += blockDim.x * 8) {
        float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int d = 0; d < nope; ++d) {
            unsigned int w = *(const unsigned int*)(kvb + (rbase + d) * rb + (col0 >> 1));
            float qs = qp[d] * sc[rbase + d];
#pragma unroll
            for (int k = 0; k < 8; ++k)
                acc[k] += qs * (float)((int)((w >> (4 * k)) & 0xFu) - 8);
        }
#pragma unroll
        for (int k = 0; k < 8; ++k) qabs[(size_t)head * kvl + col0 + k] = acc[k];
    }
}

#define CK(e) do { hipError_t _e=(e); if(_e!=hipSuccess){ \
    printf("HIP error %s at line %d\n", hipGetErrorString(_e), __LINE__); exit(1);} } while(0)

int main() {
    const int H = 64, kvl = 512, nope = 128, vh = 128, qh = 192;
    const size_t rows = (size_t)H * (nope + vh), rb = kvl / 2;
    const size_t kvb_bytes = rows * rb;
    printf("MLA kv_b: H=%d kvl=%d nope=%d vh=%d -> %.2f MB int4/layer (78 layers/token)\n",
           H, kvl, nope, vh, kvb_bytes / 1048576.0);

    unsigned char* kvb; float *sc, *q, *clat, *qabs, *ctx;
    CK(hipMalloc(&kvb, kvb_bytes)); CK(hipMalloc(&sc, rows * sizeof(float)));
    CK(hipMalloc(&q, (size_t)H * qh * sizeof(float)));
    CK(hipMalloc(&clat, (size_t)H * kvl * sizeof(float)));
    CK(hipMalloc(&qabs, (size_t)H * kvl * sizeof(float)));
    CK(hipMalloc(&ctx, (size_t)H * vh * sizeof(float)));
    CK(hipMemset(kvb, 0x53, kvb_bytes)); CK(hipMemset(sc, 0x3c, rows * sizeof(float)));
    CK(hipMemset(q, 0x3c, (size_t)H * qh * sizeof(float)));
    CK(hipMemset(clat, 0x3c, (size_t)H * kvl * sizeof(float)));

    const int REPS = 200;
    hipEvent_t t0, t1; CK(hipEventCreate(&t0)); CK(hipEventCreate(&t1));
    float ms;

    printf("\nmla_value (contracts kvl=%d over H*vh=%d rows):\n", kvl, H * vh);
    {
        dim3 gs((H * vh + 255) / 256), bs(256);
        dim3 gw((H * vh + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK), bw(ROWS_PER_BLOCK * WAVE);
        double base = 0;
        for (int v = 0; v < 3; ++v) {
            for (int w = 0; w < 3; ++w) {
                if (v==0) value_shipped<<<gs,bs>>>(clat,kvb,sc,H,nope,vh,kvl,ctx);
                if (v==1) value_wave<<<gw,bw>>>(clat,kvb,sc,H,nope,vh,kvl,ctx);
                if (v==2) value_wave_v4<<<gw,bw>>>(clat,kvb,sc,H,nope,vh,kvl,ctx);
            }
            CK(hipDeviceSynchronize()); CK(hipEventRecord(t0));
            for (int r = 0; r < REPS; ++r) {
                if (v==0) value_shipped<<<gs,bs>>>(clat,kvb,sc,H,nope,vh,kvl,ctx);
                if (v==1) value_wave<<<gw,bw>>>(clat,kvb,sc,H,nope,vh,kvl,ctx);
                if (v==2) value_wave_v4<<<gw,bw>>>(clat,kvb,sc,H,nope,vh,kvl,ctx);
            }
            CK(hipEventRecord(t1)); CK(hipEventSynchronize(t1)); CK(hipEventElapsedTime(&ms,t0,t1));
            double per = ms / REPS; if (v==0) base = per;
            const char* n[3] = {"A shipped (thread-per-row, 32 lines/load)",
                                "B wave-per-row (D3, byte loads)",
                                "C wave-per-row + uint4 loads"};
            printf("  %-42s %7.4f ms  %5.2fx  -> %5.1f ms/token\n", n[v], per, base/per, per*78);
        }
    }

    printf("\nmla_absorb (contracts nope=%d into kvl=%d columns):\n", nope, kvl);
    {
        dim3 gs((H * kvl + 255) / 256), bs(256);
        double base = 0;
        for (int v = 0; v < 2; ++v) {
            for (int w = 0; w < 3; ++w) {
                if (v==0) absorb_shipped<<<gs,bs>>>(q,kvb,sc,H,qh,nope,vh,kvl,qabs);
                if (v==1) absorb_vec<<<dim3(H),dim3(64)>>>(q,kvb,sc,H,qh,nope,vh,kvl,qabs);
            }
            CK(hipDeviceSynchronize()); CK(hipEventRecord(t0));
            for (int r = 0; r < REPS; ++r) {
                if (v==0) absorb_shipped<<<gs,bs>>>(q,kvb,sc,H,qh,nope,vh,kvl,qabs);
                if (v==1) absorb_vec<<<dim3(H),dim3(64)>>>(q,kvb,sc,H,qh,nope,vh,kvl,qabs);
            }
            CK(hipEventRecord(t1)); CK(hipEventSynchronize(t1)); CK(hipEventElapsedTime(&ms,t0,t1));
            double per = ms / REPS; if (v==0) base = per;
            const char* n[2] = {"A shipped (thread-per-column, 16B/load)",
                                "B uint loads, 8 cols/thread"};
            printf("  %-42s %7.4f ms  %5.2fx  -> %5.1f ms/token\n", n[v], per, base/per, per*78);
        }
    }
    return 0;
}
