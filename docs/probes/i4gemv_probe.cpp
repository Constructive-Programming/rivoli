// Probe: how much of LPDDR5X peak does the int4 GEMV inner loop actually reach,
// and how much does vectorizing the packed-weight load buy?
//
// The shipped kernel (moe_fused.hip::dot_i4_wave, linalg.hip::gemv_i4) has lane
// `l` read ONE BYTE per iteration at column i ≡ l (mod 32). A wave therefore
// touches only 16 consecutive bytes per load instruction — 1/8th of a 128 B
// cache line — so the loop issues 8x more loads than the traffic needs and keeps
// very few bytes in flight. This probe compares that against loading 4 B (uint,
// wave = 128 B = exactly one line) and 16 B (uint4, wave = 512 B = four lines)
// per lane, on the real GLM-5.2 MoE shapes.
//
// Shapes: one sparse layer's routed batch, E=9 experts (top-8 + shared),
// hidden=6144, moe_inter=2048 — 170 MB of int4, far past any cache, so the
// measured rate is honest DRAM streaming.
//
// build: hipcc -O3 --offload-arch=gfx1151 i4gemv_probe.cpp -o i4gemv_probe

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#define WAVE 32
#define ROWS_PER_BLOCK 8

__device__ __forceinline__ float wave_sum(float v) {
    for (int o = WAVE / 2; o > 0; o >>= 1) v += __shfl_down(v, o, WAVE);
    return v;
}

// ---- A) shipped form: one byte per lane per step (16 B per wave-load) --------
__device__ __forceinline__ float dot_i4_b1(const float* __restrict__ v,
                                           const unsigned char* __restrict__ row,
                                           int dim, int lane) {
    float acc = 0.0f;
    for (int i = lane; i < dim; i += WAVE) {
        unsigned char b = row[i >> 1];
        int n = (i & 1) ? (b >> 4) : (b & 0x0F);
        acc += v[i] * (float)(n - 8);
    }
    return wave_sum(acc);
}

// ---- B) uint load: 4 B = 8 nibbles per lane (128 B per wave-load = 1 line) ---
// Lane l owns columns [base + l*8, +8). Requires dim % (WAVE*8) == 0 and the row
// base 4 B-aligned; both hold for every GLM dim (6144, 2048, 12288, 512).
__device__ __forceinline__ float dot_i4_u32(const float* __restrict__ v,
                                            const unsigned char* __restrict__ row,
                                            int dim, int lane) {
    float acc = 0.0f;
    const unsigned int* rw = (const unsigned int*)row;
    for (int base = 0; base < dim; base += WAVE * 8) {
        int col = base + lane * 8;
        unsigned int w = rw[col >> 3];   // 8 nibbles
        const float* vv = v + col;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            int n = (int)((w >> (4 * k)) & 0xFu);
            acc += vv[k] * (float)(n - 8);
        }
    }
    return wave_sum(acc);
}

// ---- C) uint4 load: 16 B = 32 nibbles per lane (512 B per wave-load) ---------
// Lane l owns columns [base + l*32, +32). Requires dim % (WAVE*32) == 0 and the
// row base 16 B-aligned. 6144 and 2048 both satisfy the dim condition.
__device__ __forceinline__ float dot_i4_u128(const float* __restrict__ v,
                                             const unsigned char* __restrict__ row,
                                             int dim, int lane) {
    float acc = 0.0f;
    for (int base = 0; base < dim; base += WAVE * 32) {
        int col = base + lane * 32;
        uint4 w = *(const uint4*)(row + (col >> 1));
        const float* vv = v + col;
        unsigned int ws[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
        for (int q = 0; q < 4; ++q) {
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                int n = (int)((ws[q] >> (4 * k)) & 0xFu);
                acc += vv[q * 8 + k] * (float)(n - 8);
            }
        }
    }
    return wave_sum(acc);
}

// Generic gate/up pass parameterised on the dot variant (VARIANT: 0=b1,1=u32,2=u128).
template <int VARIANT>
__global__ void gateup(const float* __restrict__ x, int hidden, int inter, int E,
                       const unsigned char* __restrict__ gate,
                       const unsigned char* __restrict__ up,
                       const float* __restrict__ gscale, const float* __restrict__ uscale,
                       float* __restrict__ h_out) {
    int lane = threadIdx.x & (WAVE - 1);
    size_t r = (size_t)blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / WAVE;
    if (r >= (size_t)E * inter) return;
    size_t rb = (size_t)(hidden + 1) / 2;
    const unsigned char* grow = gate + r * rb;
    const unsigned char* urow = up + r * rb;
    float g, u;
    if (VARIANT == 0)      { g = dot_i4_b1(x, grow, hidden, lane);   u = dot_i4_b1(x, urow, hidden, lane); }
    else if (VARIANT == 1) { g = dot_i4_u32(x, grow, hidden, lane);  u = dot_i4_u32(x, urow, hidden, lane); }
    else                   { g = dot_i4_u128(x, grow, hidden, lane); u = dot_i4_u128(x, urow, hidden, lane); }
    if (lane == 0) {
        float gs = g * gscale[r];
        h_out[r] = (gs / (1.0f + __expf(-gs))) * (u * uscale[r]);
    }
}

// down pass: rows over `hidden`, contracting `inter`.
template <int VARIANT>
__global__ void down(int hidden, int inter, int E,
                     const unsigned char* __restrict__ dw, const float* __restrict__ dscale,
                     const float* __restrict__ h_in, float* __restrict__ partial) {
    int lane = threadIdx.x & (WAVE - 1);
    size_t r = (size_t)blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / WAVE;
    if (r >= (size_t)E * hidden) return;
    int e = (int)(r / hidden);
    size_t rb = (size_t)(inter + 1) / 2;
    const unsigned char* row = dw + r * rb;
    const float* he = h_in + (size_t)e * inter;
    float d;
    if (VARIANT == 0)      d = dot_i4_b1(he, row, inter, lane);
    else if (VARIANT == 1) d = dot_i4_u32(he, row, inter, lane);
    else                   d = dot_i4_u128(he, row, inter, lane);
    if (lane == 0) partial[r] = d * dscale[r];
}

#define CK(e) do { hipError_t _e=(e); if(_e!=hipSuccess){ \
    printf("HIP error %s at line %d\n", hipGetErrorString(_e), __LINE__); exit(1);} } while(0)

int main() {
    const int hidden = 6144, inter = 2048, E = 9;
    const size_t rb_h = hidden / 2, rb_i = inter / 2;
    const size_t n_gu = (size_t)E * inter;      // gate/up rows
    const size_t n_dn = (size_t)E * hidden;     // down rows
    const size_t gu_bytes = n_gu * rb_h;        // per projection
    const size_t dn_bytes = n_dn * rb_i;
    const size_t total = 2 * gu_bytes + dn_bytes;
    printf("one MoE layer batch: E=%d hidden=%d inter=%d -> %.1f MB int4\n",
           E, hidden, inter, total / 1048576.0);

    unsigned char *gate, *up, *dw;
    float *x, *gs, *us, *ds, *h, *part;
    CK(hipMalloc(&gate, gu_bytes)); CK(hipMalloc(&up, gu_bytes)); CK(hipMalloc(&dw, dn_bytes));
    CK(hipMalloc(&x, hidden * sizeof(float)));
    CK(hipMalloc(&gs, n_gu * sizeof(float))); CK(hipMalloc(&us, n_gu * sizeof(float)));
    CK(hipMalloc(&ds, n_dn * sizeof(float)));
    CK(hipMalloc(&h, n_gu * sizeof(float))); CK(hipMalloc(&part, n_dn * sizeof(float)));
    // Content is irrelevant to timing; fill with a fixed pattern so nothing is denormal.
    CK(hipMemset(gate, 0x53, gu_bytes)); CK(hipMemset(up, 0x35, gu_bytes));
    CK(hipMemset(dw, 0x71, dn_bytes));
    CK(hipMemset(x, 0x3c, hidden * sizeof(float)));
    CK(hipMemset(gs, 0x3c, n_gu * sizeof(float))); CK(hipMemset(us, 0x3c, n_gu * sizeof(float)));
    CK(hipMemset(ds, 0x3c, n_dn * sizeof(float)));

    dim3 bgu((n_gu + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK), bdn((n_dn + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
    dim3 blk(ROWS_PER_BLOCK * WAVE);
    const int REPS = 20;
    const char* names[3] = {"A byte  (shipped, 16B/wave-load)",
                            "B uint  (4B/lane, 128B/wave-load)",
                            "C uint4 (16B/lane, 512B/wave-load)"};
    double base_ms = 0.0;

    for (int v = 0; v < 3; ++v) {
        // warm up + time
        for (int w = 0; w < 2; ++w) {
            if (v==0){ gateup<0><<<bgu,blk>>>(x,hidden,inter,E,gate,up,gs,us,h); down<0><<<bdn,blk>>>(hidden,inter,E,dw,ds,h,part);}
            if (v==1){ gateup<1><<<bgu,blk>>>(x,hidden,inter,E,gate,up,gs,us,h); down<1><<<bdn,blk>>>(hidden,inter,E,dw,ds,h,part);}
            if (v==2){ gateup<2><<<bgu,blk>>>(x,hidden,inter,E,gate,up,gs,us,h); down<2><<<bdn,blk>>>(hidden,inter,E,dw,ds,h,part);}
        }
        CK(hipDeviceSynchronize());
        hipEvent_t t0, t1; CK(hipEventCreate(&t0)); CK(hipEventCreate(&t1));
        CK(hipEventRecord(t0));
        for (int r = 0; r < REPS; ++r) {
            if (v==0){ gateup<0><<<bgu,blk>>>(x,hidden,inter,E,gate,up,gs,us,h); down<0><<<bdn,blk>>>(hidden,inter,E,dw,ds,h,part);}
            if (v==1){ gateup<1><<<bgu,blk>>>(x,hidden,inter,E,gate,up,gs,us,h); down<1><<<bdn,blk>>>(hidden,inter,E,dw,ds,h,part);}
            if (v==2){ gateup<2><<<bgu,blk>>>(x,hidden,inter,E,gate,up,gs,us,h); down<2><<<bdn,blk>>>(hidden,inter,E,dw,ds,h,part);}
        }
        CK(hipEventRecord(t1)); CK(hipEventSynchronize(t1));
        float ms; CK(hipEventElapsedTime(&ms, t0, t1));
        double per = ms / REPS;
        double gbs = total / (per * 1e-3) / 1e9;
        if (v == 0) base_ms = per;
        printf("  %-36s %7.3f ms/layer  %6.1f GB/s  %5.2fx  -> %6.0f ms/token (75 layers)\n",
               names[v], per, gbs, base_ms / per, per * 75);
    }
    printf("\n(LPDDR5X peak ~230 GB/s; 75 sparse layers/token at these shapes.)\n");
    return 0;
}
