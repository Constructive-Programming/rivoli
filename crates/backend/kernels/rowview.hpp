// The BY-VALUE argument bundles the wave-cooperative dot products take, and the one
// aliasing argument that covers all of them. Split out of common.hpp 2026-08-15 alongside
// the per-format dot headers; nothing here existed before that day, so nothing travelled.
//
// WHY THESE EXIST. The dots were positional-argument functions of 5-10 scalars and
// pointers (`fp8_dot_strided_r` took ten). Every one of them is really three things — a
// view of the ACTIVATION rows, a view of the packed WEIGHT row, and where the wave's
// lanes and results go — so the bundles below are those three, and each dot header adds
// the weight view its own format needs.
//
// ── THE ALIASING RECORD (read this before moving another pointer into a struct) ──────
//
// A pointer written `T* __restrict__ p` as a PARAMETER promises the compiler that nothing
// else in the function reaches the same object. Written as a struct MEMBER the promise is
// weaker and less portable: C99 gives restrict-qualified members meaning only through the
// containing object, and clang honours them in many cases but not as a documented
// contract. Moving a hot pointer from a parameter into a struct therefore RISKS losing a
// no-alias fact the loop was relying on to keep a load hoisted out of its body.
//
// **This refactor was accepted with that risk UNMEASURED (owner, 2026-08-15) and the job
// was to MINIMIZE it.** Three rules did that, and they are the reason the shapes below
// look the way they do:
//
//  1. **The STORE side never enters a struct.** In every one of these loops the only
//     memory WRITE is the accumulator; the weight, scale, LUT and activation spans are
//     read-only. A `__restrict__` on the store pointer alone already tells LLVM that the
//     stores cannot alias any of the loads, which is the entire fact these loops need —
//     aliasing *among* read-only spans blocks nothing, because there is no write to
//     reorder them against. So `Acc<R>` is RETURNED BY VALUE where the accumulator is
//     pure output, and passed by REFERENCE to a caller local where it is in/out. Both
//     forms are strictly stronger than the `float* __restrict__` they replace: after the
//     `__forceinline__` every one of these functions carries, the local is SROA'd into
//     registers and there is no pointer left to alias anything.
//  2. **Read-only spans go in, and are still marked `__restrict__`.** The qualifier is
//     kept on the members even though its guarantee is weaker there — it costs nothing,
//     it documents the intent, and clang does act on it in many cases.
//  3. **Everything is passed BY VALUE.** A `const RowsView&` would reintroduce a pointer
//     to a struct whose members the loop reads, which is the aliasing question again one
//     level up.
//
// Which pointers actually lost parameter-position `__restrict__`, and where they are hot:
//
//   | pointer                    | now a member of | hot in                              |
//   |----------------------------|-----------------|-------------------------------------|
//   | activation rows `x` / `v`  | `RowsView`      | every dot's inner loop (per column)  |
//   | fp8 packed row + scale row | `Fp8Row`        | `fp8_dot_strided{,_r}` dword loop    |
//   | fp8 e4m3 LUT               | `Fp8Row`        | same loop, 4 gathers per dword       |
//   | int4 packed row + scales   | `I4Row`         | `i4_dword_pass` (unroll 4)           |
//   | fp4 packed row + scales    | `F4Row`         | `f4_dword_pass` (unroll 4)           |
//   | vq index/scale/codebook    | `VqRow`         | `dot_vq_wave_r` subvector loop       |
//
// **M5's benchmarks re-price this.** The unroll depths on `fp8_dot_strided` (8),
// `i4_dword_pass` (4) and `f4_dword_pass` (4) were each chosen from a measured GB/s and a
// VGPR/occupancy read of the emitted ISA; if any of those loops regressed here it shows up
// as a bandwidth loss on exactly those arms, which is what M5 measures. Until then the
// claim in this file is "no arithmetic changed", not "no schedule changed" — the first is
// host-proven bit-identical, the second is a measurement nobody has taken.
#pragma once

#include <hip/hip_runtime.h>

// R activation rows, row-minor: row `r` starts at `x + r*stride`. At R=1 the stride is
// unread, and the single-row entry points keep a bare `const float* __restrict__`
// parameter rather than passing a view with a meaningless stride.
struct RowsView {
    const float* __restrict__ x;
    int stride;
};

// The columns one thread owns: `start`, then every `stride`-th. Wave-per-row passes
// `{lane, WAVE}`; split-K passes `{threadIdx.x, ROWS_PER_BLOCK*WAVE}` so all 256 threads
// stride coalesced. Two ints that are only ever meaningful together.
struct LaneSpan {
    int start;
    int stride;
};

// Where a two-pass dot resumes, and for which lane. Shared by the int4 and fp4 tails
// because it IS the same pair — the dword pass returns `base` and the tail continues from
// it, which is what makes the two passes a PARTITION of the columns rather than two
// independent loops each deciding its own bound.
struct TailSpan {
    int base;
    int lane;
};

// R accumulators, one per activation row. A VALUE, not a `float*`: see rule 1 above.
// Returned from the dots that own their accumulator, passed by reference to the in/out
// helpers that fold into a caller's.
template <int R>
struct Acc {
    float v[R];
};

// The two independent FMA chains a dword fast path keeps. `a0` takes the low four columns
// of each dword and `a1` the high four, so the 8 columns split into two chains that issue
// without waiting on each other; the scalar tails fold into `a0` only, because there is no
// second chain there to feed.
template <int R>
struct AccPair {
    Acc<R> a0, a1;
};

// log2 of a power-of-two fp8 scale-tile size. Every launcher taking a `block`
// REJECTS a non-power-of-two (arg guard 1003), so the hot loops index the block scale
// with a shift instead of a divide. `block` is a runtime int, and LLVM does
// strength-reduce `i / block` to a magic multiply — but the SIGNED quotient correction
// survives it, and that is what cost: measured 44 → 29 VALU per iteration in
// gemv_fp8_splitk, around 5 ops of real math, with the memory ops unchanged at 7.
// Bit-identical: same index, same order. See benchmarks.md, "Read the ISA before you
// book the device", for why grepping for `v_rcp_iflag_f32` finds nothing here.
//
// It lives with the views rather than with the fp8 dot because both sides of a row view
// use it: the dot indexes `scale[i >> bsh]` INSIDE the loop, and `linalg.hip`/`mla.hip`
// use it OUTSIDE to pick the scale row a `Fp8Row` points at (`scale + (o >> blk_shift) *
// sc_cols`). One definition, one place, both callers.
__device__ __forceinline__ int blk_shift(int block) { return 31 - __clz(block); }
