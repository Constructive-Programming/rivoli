//! The two GLU activations two expert paths each share, so one definition keeps them bit
//! for bit in step. Port of `common.hpp`'s activation half at tag `archive/hip-kernels`.

use core::intrinsics::{exp, fabs, fmuladd};

use super::formats::rbf16;
use super::llvm::{maxnum, minnum, rcp};

/// V4 `Expert.forward`'s clamped SwiGLU, before the routing weight and the bf16 store:
/// `silu(min(bf16(g), L)) · clamp(bf16(u), ±L)`, returned UNROUNDED. Four load-bearing
/// details, each a named oracle defect (argued in the tag's `common.hpp`): bf16 BEFORE the
/// clamp; the clamp is ASYMMETRIC (gate from above only); `x·sigmoid(x)`, not
/// `activation.hip::swiglu`'s division form; no rounding here, the callers round at
/// different points. `limit > 0` (both launchers refuse otherwise).
#[inline(always)]
pub fn swiglu_clamped(g: f32, u: f32, limit: f32) -> f32 {
    let ut = minnum(maxnum(rbf16(u), -limit), limit);
    let gt = minnum(rbf16(g), limit);
    gt * (1.0 / (1.0 + exp(-gt))) * ut
}

/// ocml's `tanhf` as hipcc inlines it from `/usr/lib/amdgcn/bitcode/ocml.bc` (ROCm clang
/// 23.0.0git), transliterated from the IR: an odd polynomial below |x| = 0.625, else
/// `1 − 2/(e^{2|x|} + 1)` through the 1-ulp `rcp`. PINNED to that device-lib build — a ROCm
/// upgrade can move the C++ side, and the bit-identity sweep is what would notice.
#[inline(always)]
fn tanh_ocml(x: f32) -> f32 {
    let a = fabs(x);
    let z = if a < 0.625 {
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

/// Kimi-K3's SiTU-GLU: `b1·tanh(g/b1)·sigmoid(g) · b2·tanh(u/b2)`, returned UNROUNDED.
/// NOT `swiglu_clamped` with other constants (the tag's `common.hpp` argues why): the
/// sigmoid takes the UNCAPPED gate, `up` is transformed rather than clamped, and neither operand is
/// bf16-rounded. Sigmoid in the division form, as `torch.sigmoid`.
#[inline(always)]
pub fn situ_glu(g: f32, u: f32, b1: f32, b2: f32) -> f32 {
    let a = b1 * tanh_ocml(g / b1) * (1.0 / (1.0 + exp(-g)));
    a * (b2 * tanh_ocml(u / b2))
}
