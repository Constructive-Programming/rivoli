// ── OCP MX FP4 (DeepSeek-V4-Flash's native routed experts) ──────────────────────
//
// e2m1 nibbles with ONE e8m0 (bare power-of-two) scale per F4_GROUP weights along the
// input dim. Split out of common.hpp 2026-08-15 with the other three dot families; the
// bodies and the measurements in their comments travelled verbatim, and the signatures
// were bundled into the views in `rowview.hpp` the same day (that file carries the
// aliasing argument for every pointer below).
//
// Three things separate this from the int4 block in `dot_i4.hpp` and every one of them
// is silent-wrong if it drifts:
//
//   * the codebook is NON-UNIFORM ({0,.5,1,1.5,2,3,4,6}), not `nibble − 8`, so an int4
//     decode of fp4 bytes produces plausible magnitudes and wrong text;
//   * the group is 32, not 128 — a group stride read from I4_GROUP mis-scales 3 of every
//     4 groups while leaving the first one right, which is exactly the kind of bug that
//     looks like "mostly working";
//   * the scale grid is [o_dim, ceil(K/32)] row-major, one scale ROW per output row,
//     where fp8's is a 128×128 TILE grid shared by 128 output rows.
//
// MUST match quant.rs (F4_GROUP / F4_LUT / e8m0) and v4oracle::numerics (e2m1_decode /
// e8m0_decode); the decoders themselves are `e2m1f`/`e8m0f` in formats.hpp. The oracle
// is the reference; these two are independently written from the same format
// definition, which is the point — see v4oracle/numerics.rs's module doc.
#pragma once

#include <hip/hip_runtime.h>

#include "formats.hpp"
#include "reduce.hpp"
#include "rowview.hpp"

#define F4_GROUP 32
#define F4_GROUP_SHIFT 5
static_assert((1 << F4_GROUP_SHIFT) == F4_GROUP, "F4_GROUP_SHIFT must be log2(F4_GROUP)");
static_assert(F4_GROUP % 8 == 0, "the dword fast path's 8 columns must not straddle a group");

// e8m0 scale byte → f32, `2^(b − 127)` (matches v4oracle::numerics::e8m0_decode). It lives
// HERE rather than beside its e2m1 partner in formats.hpp — moved 2026-08-15 — because
// this is the GROUP-SCALE decoder and its only two callers are the two loops below, while
// e2m1 is an element codec the indexer's activation round trip needs as well.
//
// Both endpoints are spelled out rather than left to the exponent-field shift. 0xff is the
// format's NaN, which is the right DECODE — reading it as `2^128` would be worse — but it
// does not poison anything downstream: `moe_fixed`'s saturating clamp launders a NaN into a
// finite ±2^14 (`fminf`/`fmaxf` return the non-NaN operand). So a 0xff scale byte must be
// REJECTED AT LOAD, in S3, alongside the `tid2eid` range check that `parse_tid2eid`
// performs host-side. (That obligation used to be stated at `moe.hip::moe_gate_v4` too;
// the device router was DELETED 2026-08-09 — `f4gpu.rs::route_row` carries why — so this
// is now the only place in `kernels/` that names it, which is the reason to keep it here
// rather than fold it into the deleted kernel's note.)
// b == 0 is 2^-127, which is BELOW f32's smallest normal, so `b << 23` would silently hand
// back +0 — a whole 32-weight group zeroed with no error anywhere.
//
// These bytes come off the CHECKPOINT (`<proj>.scale`, copied verbatim by the `.f4`
// repack), not from `fast_round_scale`, so "our quantizer cannot emit them" is not an
// argument that applies here: the FORMAT permits both, and what the converter happened to
// produce is not a property this decoder gets to assume.
//
// Branchless since 2026-08-08 (the M3a kernel-rate work): the two-`if` form compiled to an
// exec-mask branch region inside the fp4 dot loop's every iteration. The selects below are
// the same three cases — `umax` against the b == 0 subnormal is exact because every other
// `b<<23` is strictly larger, and the 0xff arm ORs the quiet bit onto what is then
// 0x7f800000, giving the identical 0x7fc00000 NaN. All 256 bytes are swept bitwise against
// `v4oracle::numerics::e8m0_decode` by
// `tests/f4_kernel.rs::the_branchless_decodes_match_the_oracle_bitwise`.
__device__ __forceinline__ float e8m0f(unsigned char b) {
    unsigned int t = (unsigned int)b << 23;
    unsigned int bits = t > (1u << 22) ? t : (1u << 22);
    bits |= (b == 0xff) ? (1u << 22) : 0u;
    return __uint_as_float(bits);
}

// A GLOBAL-address-space (AMDGCN AS1) byte pointer, for spans whose globalness clang
// cannot infer. A pointer that arrives as a KERNEL ARGUMENT is promoted to AS1 by
// AMDGPUPromoteKernelArguments (that is why the fp4 dot's activation loads lower to
// `global_load_b128`), but a pointer LOADED from device memory — the six spans in an
// `ExpertDescF4` — reaches the loads as a generic value, and the M3a ISA read
// (docs/investigations/v4-decode-decomposition.md) measured the consequence: the weight
// dword and scale byte lower to `flat_load_*`, which on gfx11 take the slower path, count
// against `lgkmcnt` too, and put a mid-body `s_waitcnt` in an issue-bound loop. Typing the
// span AS1 at the source level is the fix that cannot be optimized away — a round-trip
// `(T*)(AS1 T*)p` cast is folded by instcombine, and an
// `assume(!is_shared && !is_private)` was measured not to reach InferAddressSpaces either
// (both tried 2026-08-08; the loads stayed flat). The generic→AS1 cast is a no-op on
// gfx11 (the representations coincide) and asserts only the address SPACE, nothing about
// aliasing.
typedef const unsigned char __attribute__((address_space(1)))* gu8p;

// One packed fp4 weight row and its e8m0 scale row:
//   out[t] = Σ_i v[t·stride + i] · e2m1(nibble(i)) · e8m0(scale[i / F4_GROUP])
// `w` = packed[o·(dim/2)..], `scale` = scale + o·ceil(dim/F4_GROUP).
//
// Both spans are `gu8p` — AS1-typed, see the typedef above — because both come out of an
// `ExpertDescF4` at every call site and the flat_load they otherwise lower to is a
// measured cost on an issue-bound loop. The address space and the `__restrict__` are
// orthogonal: AS1 asserts WHERE the bytes live, `__restrict__` asserts that nothing else
// reaches them.
struct F4Row {
    gu8p __restrict__ w;
    gu8p __restrict__ scale;
    int dim;
};

// `dot_f4_wave_r`'s dword fast path, returning the column `base` its scalar tail resumes
// from — the fp4 twin of `dot_i4.hpp::i4_dword_pass`, and lifted out on the same day and
// for the same reason (SHAPE ONLY; the pragma, the fold order and every measured number
// below travel with the loop).
//
// Same two-path shape as `dot_i4_wave_r` — a dword (8 nibbles = 8 consecutive columns) per
// lane when the row is 4-byte aligned, a scalar tail otherwise — for the same reason: 8
// columns starting at a multiple of 8 always share ONE group scale, so the scale decode
// amortises 8×. The nibble ORDER is the checked half: nibble k of the dword is column
// col+k, i.e. a byte's LOW nibble is the EVEN column, which is what `convert.py`'s
// `stack([TABLE[low], TABLE[high]]).flatten()` and `WMat::Fp4::row` both mean. Reading it
// high-first is a permutation INSIDE each scale group, so it survives every summary
// statistic — `v4oracle::Defect::Fp4NibbleSwap` is what keeps it A/B-able.
//
// NOT merged with `dot_i4_wave_r`, and the reason is arithmetic rather than caution:
// that one multiplies `v[i] · (n−8) · scale` PER ELEMENT because hoisting its f32 group
// scale would re-associate the product (see its own note, and the bit-identity constraint
// `tests/kernel.rs` pins on it). This one hoists the scale out of eight columns, which is
// legal only because e8m0 is a bare power of two and the multiply is therefore exact. A
// shared template would need a policy parameter meaning "is this scale exact?" — an
// abstraction whose two instantiations disagree at the one line it exists to share.
//
// Predicated on the WEIGHT row's 4-byte alignment, which is what the dword read needs; an
// unaligned row returns 0 and the tail takes it whole. The `float4` loads below are on the
// ACTIVATION and are NOT checked — a 16-byte-aligned `v` is an unchecked caller obligation
// (`launch_moe_expert_range_f4`'s `# Safety`), and a misaligned one faults rather than
// falling back to the tail.
template <int R>
__device__ __forceinline__ int f4_dword_pass(RowsView v, F4Row w, int lane, AccPair<R>& a) {
    int base = 0;
    if ((((size_t)w.w) & 3u) != 0) return base;
    const unsigned int __attribute__((address_space(1)))* rw =
        (const unsigned int __attribute__((address_space(1)))*)w.w;
    // Unrolled 2026-08-09 (M11). Un-pragma'd, this loop drained inside its own body
    // (`s_waitcnt vmcnt(1)`/`vmcnt(0)` in-body) — ONE iteration of loads in flight, ever,
    // which is M7's disease; `fp8_dot_strided` (dot_fp8.hpp) was fixed for it and the
    // exemption beside that pragma names `fp8_dot_strided_r`, so this loop was never in
    // that conversation. `dot_bench v4res` MICROBENCHMARK at the engine's dims: 146.63 →
    // 195.27 GB/s. Depth is what buys it — a ballast with the decode and every FMA
    // removed buys only +12.8%, so the bound was never issue rate.
    //
    // A change that grows this body must re-read the ISA. Unroll 4 is 125 VGPR on
    // `moe_gateup_f4` (**10 waves/SIMD**, down from 16) and 93 on `moe_down_f4` (still 16);
    // no spill on either, 0 scratch. Unroll 2 keeps 16 waves everywhere at 74/66 VGPR and
    // measured 172.55; **both rungs are registered arms of an engine A/B and
    // `docs/investigations/v4-decode-decomposition.md` §M11b decides between them on the
    // engine wall, not this comment.**
    //
    // Fold order must stay the ascending-`base` sum, and this was READ OUT of the
    // compiler rather than argued: the device IR carries `contract` (FMA fusion, which
    // stock already relies on) but **zero `reassoc`/`fast`/`nnan`** at stock, unroll 2 and
    // unroll 4; each accumulator stays a SINGLE serial fadd chain terminating at the loop
    // phi — exactly two float phis in the loop header at all three depths, no partial-
    // accumulator split — and the chain is `phi + t(base) + t(base+256) + …`, ascending.
    // The epilogue remainder chains off the main loop's exit values, so it continues the
    // same order. `-ffast-math` is absent (`build.rs`), but `-ffp-contract=fast` is clang's
    // HIP default, which is why this was read and not assumed
    // (`tests/f4_kernel.rs::the_fp4_dispatch_hash_pins_the_clamp_hoist` records that
    // lesson).
    //
    // Multi-trip coverage is `tests/f4_kernel.rs::the_dword_path_matches_the_oracle_at_
    // multiple_trips` (1280/1024 = 5 and 4 trips; at `V4Config::toy` gate/up runs this loop
    // ONCE and down never enters it, so every other test executes only the remainder copy).
    // Measured 2026-08-09: it gates arithmetic wrong past the first trip — red at 65x
    // tolerance on an injected `n7 = 0 when base != 0`, while all 27 other tests stay green.
    // It does NOT gate this bound (measured 2026-08-09 by building both): `<=` -> `<` moves
    // the last trip into the scalar tail below, which resumes from `base`, and the test
    // still passes — the two paths are a PARTITION, so an off-by-one that SHORTENS this
    // loop is a performance bug no test will catch. Loosening it is a different animal:
    // reading `rw[col >> 3]` past the row faults or corrupts. The remainder is unreachable
    // at the engine's dims but reachable in production in principle — the launcher guards
    // `% ACT_QUANT_BLOCK` (128), not 256, so a conforming dim like 1152 or 3712 gives an
    // odd trip count.
#pragma unroll 4
    for (; base + WAVE * 8 <= w.dim; base += WAVE * 8) {
        int col = base + lane * 8;
        unsigned int p = rw[col >> 3];                        // 8 nibbles = 8 columns
        float s = e8m0f(w.scale[col >> F4_GROUP_SHIFT]);      // one group for all 8
        // Decoded ONCE for every row — the weight side is the expensive half.
        float n0 = e2m1f(p), n1 = e2m1f(p >> 4), n2 = e2m1f(p >> 8), n3 = e2m1f(p >> 12);
        float n4 = e2m1f(p >> 16), n5 = e2m1f(p >> 20), n6 = e2m1f(p >> 24),
              n7 = e2m1f(p >> 28);
#pragma unroll
        for (int t = 0; t < R; ++t) {
            const float* vt = v.x + (size_t)t * v.stride;
            float4 x0 = *(const float4*)(vt + col);
            float4 x1 = *(const float4*)(vt + col + 4);
            a.a0.v[t] += s * (x0.x * n0 + x0.y * n1 + x0.z * n2 + x0.w * n3);
            a.a1.v[t] += s * (x1.x * n4 + x1.y * n5 + x1.z * n6 + x1.w * n7);
        }
    }
    return base;
}

// The per-column remainder, from `t.base` to `w.dim` — `i4_tail_accum`'s fp4 twin. Folds
// into `a0` only, for the same reason: `a1` is the dword path's second FMA chain and there
// is no second chain here. The scale is decoded PER COLUMN rather than hoisted, because
// this path has no group of eight to hoist it out of.
template <int R>
__device__ __forceinline__ void f4_tail_accum(RowsView v, F4Row w, TailSpan t, Acc<R>& a0) {
    for (int i = t.base + t.lane; i < w.dim; i += WAVE) {
        unsigned char b = w.w[i >> 1];
        unsigned int n = (i & 1) ? (unsigned int)(b >> 4) : (unsigned int)(b & 0x0F);
        float s = e8m0f(w.scale[i >> F4_GROUP_SHIFT]);
#pragma unroll
        for (int r = 0; r < R; ++r) a0.v[r] += v.x[(size_t)r * v.stride + i] * e2m1f(n) * s;
    }
}

// Wave-cooperative group-scaled FP4 dot, R token rows against ONE read of the weight row.
// Result on lane 0.
template <int R>
__device__ __forceinline__ Acc<R> dot_f4_wave_r(RowsView v, F4Row w, int lane) {
    AccPair<R> a;
#pragma unroll
    for (int t = 0; t < R; ++t) {
        a.a0.v[t] = 0.0f;
        a.a1.v[t] = 0.0f;
    }
    int base = f4_dword_pass<R>(v, w, lane, a);
    f4_tail_accum<R>(v, w, TailSpan{base, lane}, a.a0);
    Acc<R> out;
#pragma unroll
    for (int t = 0; t < R; ++t) out.v[t] = wave_sum(a.a0.v[t] + a.a1.v[t]);
    return out;
}
