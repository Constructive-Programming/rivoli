//! Red-proof plants: one deliberately defective helper per family, each shown adversarial on
//! the CPU first (Phase 0b's sweeps, recorded in `docs/investigations/rust-kernels.md`).
//! Everything a plant module does not replace is the real prelude.
// Each plant module re-exports its real family (`pub use …::*`) so the probes naming the
// family still resolve, and a probe set uses only part of each family: the unused rest of a
// private module's re-exports would otherwise warn, which the workspace denies.
#![allow(dead_code, unused_imports)]

// One plant per build: a red proof isolates ONE defect, and two plants that replace the same
// family (`situ`/`common`, `reduce`/`barrier`) would not even resolve.
const _: () = assert!(
    cfg!(feature = "plant_formats") as u8
        + cfg!(feature = "plant_actquant") as u8
        + cfg!(feature = "plant_common") as u8
        + cfg!(feature = "plant_situ") as u8
        + cfg!(feature = "plant_reduce") as u8
        + cfg!(feature = "plant_barrier") as u8
        + cfg!(feature = "plant_fp8") as u8
        + cfg!(feature = "plant_nib") as u8
        + cfg!(feature = "plant_vq") as u8
        <= 1,
    "enable at most one plant_* feature per build"
);

/// formats: `f2e4m3_rne`'s subnormal tie rounds half UP instead of half-to-even. Defect set:
/// the subnormal ties with an EVEN kept mantissa, `(m + 0.5)·2^-9` for m in {2, 4, 6} — six
/// inputs of 2^32 (CPU sweep). (The tempting `<= 2^-10` → `<` plant is a NO-OP: that tie
/// falls into the subnormal path and rounds to the same even zero — CPU sweep: 0 inputs.)
#[cfg(feature = "plant_formats")]
pub mod formats {
    pub use crate::prelude::formats::*;
    pub fn f2e4m3_rne(x: f32) -> u8 {
        let a = x.abs();
        let scaled = a * 512.0;
        let m = core::intrinsics::floor(scaled);
        // e4m3's subnormal band, above the round-to-zero threshold.
        let subnormal = a > 0.000_976_562_5 && a < 0.015_625;
        let tie = scaled - m == 0.5;
        if subnormal && tie {
            return (if x.is_sign_negative() { 0x80 } else { 0 }) | (m as u8 + 1); // PLANT
        }
        crate::prelude::formats::f2e4m3_rne(x)
    }
}

/// actquant: the clamp via `minnum`/`maxnum` instead of comparisons, which launders a NaN
/// quotient into a finite bound. Defect set: exactly the inputs where `x / s` is NaN.
#[cfg(feature = "plant_actquant")]
pub mod actquant {
    pub use crate::prelude::actquant::*;
    use crate::prelude::formats::{e2m1f, e4m3f, f2e2m1, f2e4m3_rne};
    use crate::prelude::llvm::{maxnum, minnum};
    pub fn act_quant_roundtrip(x: f32, s: f32) -> f32 {
        e4m3f(f2e4m3_rne(minnum(maxnum(x / s, -FP8_MAX), FP8_MAX))) * s
    }
    pub fn fp4_quant_roundtrip(x: f32, s: f32) -> f32 {
        e2m1f(f2e2m1(minnum(maxnum(x / s, -FP4_MAX), FP4_MAX))) * s
    }
}

/// common: `swiglu_clamped` clamping the gate from BOTH sides (`Defect::SwigluClampGateBothSides`).
/// Defect set: bf16(g) < −limit.
#[cfg(feature = "plant_common")]
pub mod common {
    use crate::prelude::formats::rbf16;
    use crate::prelude::llvm::{maxnum, minnum};
    pub use crate::prelude::*;
    pub fn swiglu_clamped(g: f32, u: f32, limit: f32) -> f32 {
        let ut = minnum(maxnum(rbf16(u), -limit), limit);
        let gt = minnum(maxnum(rbf16(g), -limit), limit);
        gt * (1.0 / (1.0 + core::intrinsics::exp(-gt))) * ut
    }
}

/// reduce: `wave_sum` as a xor butterfly instead of the shuffle-down ladder. Lane 0 is the
/// same sum in the same order (CPU model), every other lane differs — the harness compares
/// all 32 lanes, so this reddens only because it does.
#[cfg(feature = "plant_reduce")]
pub mod reduce {
    use crate::prelude::llvm::wave_shuffle;
    pub use crate::prelude::reduce::*;
    pub fn wave_sum(v: f32) -> f32 {
        [16u32, 8, 4, 2, 1]
            .into_iter()
            .fold(v, |v, o| v + wave_shuffle(v, lane_id() ^ o))
    }
}

// EXEMPT FROM THE DUPLICATION GATE, from here to the end of the file: the plants below are
// copies of live prelude helpers with exactly one deliberate defect each (marked `// PLANT`),
// so the gate can be shown red. Folding a plant into the helper it copies (a cfg'd line in
// the prelude) would put red-proof branches into the shipped source; sharing the unchanged
// lines would make the defect a diff against code that moves. Measured 2026-11-18 with the
// markers absent: jscpd's clones are these plants against `activation` (situ), `dot_fp8`,
// `dot_nib` and `dot_vq`, and nothing above this line.
// jscpd:ignore-start
/// situ: ocml tanh's polynomial/exponential crossover moved from |x| < 0.625 to < 0.5. Both
/// branches approximate tanh to ~1 ulp, so the defect is a few-ulp move confined to
/// |g/b1| or |u/b2| in [0.5, 0.625) — the kind of drift a ROCm device-lib change would bring.
#[cfg(feature = "plant_situ")]
pub mod common {
    use crate::prelude::llvm::rcp;
    pub use crate::prelude::*;
    use core::intrinsics::{exp, fabs, fmuladd};
    pub fn tanh_plant(x: f32) -> f32 {
        let a = fabs(x);
        let z = if a < 0.5 {
            // PLANT (0.625)
            let x2 = x * x;
            let c = f32::from_bits;
            let p = fmuladd(x2, c(0xBBBA_C73D), c(0x3CA9_08C9));
            let p = fmuladd(
                x2,
                fmuladd(x2, fmuladd(x2, p, c(0xBD5C_1C4E)), c(0x3E08_8382)),
                c(0xBEAA_AA99),
            );
            fmuladd(x2, a * p, a)
        } else {
            fmuladd(rcp(exp(a * 2.0) + 1.0), -2.0, 1.0)
        };
        z.copysign(x)
    }
    pub fn situ_glu(g: f32, u: f32, b1: f32, b2: f32) -> f32 {
        let a = b1 * tanh_plant(g / b1) * (1.0 / (1.0 + exp(-g)));
        a * (b2 * tanh_plant(u / b2))
    }
}

/// fp8: the accumulator `fma(s, quad, acc)` split into `acc + s*quad` — one missing
/// `fmuladd`, the spike's failure mode. Wave and split-K forms both route through here.
#[cfg(feature = "plant_fp8")]
pub mod dot_fp8 {
    pub use crate::prelude::dot_fp8::{FP8_SINGLE_ROW_UNROLL, Fp8Row};
    use crate::prelude::reduce::{WAVE, wave_sum};
    use crate::prelude::rowview::{LaneSpan, RowsView, blk_shift, for_strided, quad};
    use core::intrinsics::fmuladd;
    pub unsafe fn fp8_dot_strided_r<const R: usize, const U: usize>(
        x: RowsView,
        w: Fp8Row,
        span: LaneSpan,
    ) -> [f32; R] {
        let mut acc = [0.0f32; R];
        let n4 = if w.block >= 4 { w.i_dim >> 2 } else { 0 };
        let bsh = blk_shift(w.block) as u32;
        unsafe {
            let lut = |b: u32| *w.lut.add((b & 0xff) as usize);
            for_strided::<U>(span.start, n4, span.stride, |j| {
                let p = *w.w.cast::<u32>().add(j);
                let i0 = j << 2;
                let s = *w.scale.add(i0 >> bsh);
                let l = [lut(p), lut(p >> 8), lut(p >> 16), lut(p >> 24)];
                for (r, a) in acc.iter_mut().enumerate() {
                    let xr = x.x.add(r * x.stride + i0);
                    *a += s * quad([*xr, *xr.add(1), *xr.add(2), *xr.add(3)], l); // PLANT
                }
            });
            for_strided::<1>((n4 << 2) + span.start, w.i_dim, span.stride, |i| {
                let l = *w.lut.add(usize::from(*w.w.add(i)));
                let s = *w.scale.add(i >> bsh);
                for (r, a) in acc.iter_mut().enumerate() {
                    *a = fmuladd(*x.x.add(r * x.stride + i) * l, s, *a);
                }
            });
        }
        acc
    }
    pub unsafe fn dot_fp8_wave_r<const R: usize, const U: usize>(
        x: RowsView,
        w: Fp8Row,
        lane: usize,
    ) -> [f32; R] {
        unsafe {
            fp8_dot_strided_r::<R, U>(
                x,
                w,
                LaneSpan {
                    start: lane,
                    stride: WAVE,
                },
            )
        }
        .map(wave_sum)
    }
}

/// nib: the tail re-associated as `x·(n·s)` — the hazard `dot_i4.hpp` names. Defect set: the
/// columns the TAIL owns; a dim fully covered by the dword pass (aligned rows, dim % 256 == 0)
/// must stay green.
#[cfg(feature = "plant_nib")]
pub mod dot_nib {
    pub use crate::prelude::dot_nib::*;
    use crate::prelude::reduce::{WAVE, wave_sum};
    use crate::prelude::rowview::{F32x4, RowsView, for_strided, quad};
    use core::intrinsics::fmuladd;
    pub unsafe fn dot_nib_wave_r<const R: usize, const U: usize, F: NibFormat>(
        x: RowsView,
        w: NibRow<F>,
        lane: usize,
    ) -> [f32; R] {
        let (mut a0, mut a1) = ([0.0f32; R], [0.0f32; R]);
        let base = if w.w as usize & 3 == 0 {
            w.dim / (WAVE * 8) * (WAVE * 8)
        } else {
            0
        };
        unsafe {
            let scale = |col: usize| F::scale(*w.scale.add(col >> F::GROUP_SHIFT));
            for_strided::<U>(0, base, WAVE * 8, |step| {
                let col = step + lane * 8;
                let p = *w.w.cast::<u32>().add(col >> 3);
                let s = scale(col);
                let n_lo: [f32; 4] = core::array::from_fn(|k| F::decode(p >> (4 * k)));
                let n_hi: [f32; 4] = core::array::from_fn(|k| F::decode(p >> (4 * k + 16)));
                for (r, (lo, hi)) in a0.iter_mut().zip(a1.iter_mut()).enumerate() {
                    let xr = x.at(r, col);
                    *lo = fmuladd(s, quad(F32x4::load(xr).0, n_lo), *lo);
                    *hi = fmuladd(s, quad(F32x4::load(xr.add(4)).0, n_hi), *hi);
                }
            });
            for_strided::<1>(base + lane, w.dim, WAVE, |i| {
                let n = F::decode(u32::from(*w.w.add(i >> 1)) >> (4 * (i & 1)));
                let s = scale(i);
                for (r, a) in a0.iter_mut().enumerate() {
                    *a = fmuladd(*x.at(r, i), n * s, *a); // PLANT
                }
            });
        }
        core::array::from_fn(|r| wave_sum(a0[r] + a1[r]))
    }
}

/// vq: the subvector dot contracted the other way round, `fma(x1, c1, x0·c0)` — the RIGHT
/// product fused instead of the left, i.e. a misread of hipcc's ISA.
#[cfg(feature = "plant_vq")]
pub mod dot_vq {
    pub use crate::prelude::dot_vq::*;
    use crate::prelude::formats::bf16f;
    use crate::prelude::reduce::{WAVE, wave_sum};
    use crate::prelude::rowview::{F32x4, RowsView, for_strided};
    use core::intrinsics::fmuladd;
    #[derive(Clone, Copy)]
    #[repr(C, align(4))]
    struct F16x2([f16; 2]);
    pub unsafe fn dot_vq_wave_r<const R: usize>(x: RowsView, w: VqRow, lane: usize) -> [f32; R] {
        let mut acc = [0.0f32; R];
        unsafe {
            for_strided::<1>(lane, w.i_dim / VQ_DIM, WAVE, |t| {
                let bitpos = t * VQ_INDEX_BITS;
                let raw = u32::from(*w.idx.add(bitpos >> 3))
                    | (u32::from(*w.idx.add((bitpos >> 3) + 1)) << 8);
                let c =
                    w.cb.add((((raw >> (bitpos & 7)) & 0xFFF) as usize) * VQ_DIM);
                let (c01, c23) = (*c.cast::<F16x2>(), *c.add(2).cast::<F16x2>());
                let cv = [
                    c01.0[0] as f32,
                    c01.0[1] as f32,
                    c23.0[0] as f32,
                    c23.0[1] as f32,
                ];
                let s = bf16f(*w.scale.add(t / VQ_SUBS_PER_GROUP));
                for (r, a) in acc.iter_mut().enumerate() {
                    let xv = F32x4::load(x.x.add(r * x.stride + t * VQ_DIM)).0;
                    let q = fmuladd(
                        xv[3],
                        cv[3],
                        fmuladd(xv[2], cv[2], fmuladd(xv[1], cv[1], xv[0] * cv[0])),
                    ); // PLANT
                    *a = fmuladd(s, q, *a);
                }
            });
        }
        acc.map(wave_sum)
    }
}

/// barrier: `block_sum_lds` WITHOUT its trailing barrier. A race, so a bit comparison may or
/// may not see it on a given run; its standing red proof is the ISA census (`s_barrier` per
/// kernel), which must go DIFF on `k_block_sum2`.
#[cfg(feature = "plant_barrier")]
pub mod reduce {
    pub use crate::prelude::reduce::*;
    pub unsafe fn block_sum_lds(v: f32, red: *mut f32, tid: usize, block_dim: usize) -> f32 {
        unsafe {
            *red.add(tid) = v;
            sync_threads();
            let mut o = block_dim >> 1;
            while o > 0 {
                if tid < o {
                    *red.add(tid) += *red.add(tid + o);
                }
                sync_threads();
                o >>= 1;
            }
            *red // PLANT: no trailing sync_threads()
        }
    }
}
// jscpd:ignore-end
