//! The V4-Flash numeric primitives, transliterated from `inference/kernel.py`.
//!
//! Every one of these is a *silent-wrong* risk on the GPU: a wrong rounding mode, a wrong
//! block size, a wrong scale rounding — none of them crash, all of them produce fluent
//! wrong text. They are separated from `forward.rs` because they are the layer that can be
//! tested exhaustively (256 fp8 patterns, 16 fp4 patterns, 65536 bf16 patterns) rather than
//! only against a golden.
//!
//! **Scope of fidelity.** The reference stores activations in bf16 and quantizes them to
//! fp8/fp4 inside every GEMM. This module reproduces the *values* that arithmetic produces,
//! not its accumulation order: `fp8_gemm`'s two-level fp32 accumulator and `sparse_attn`'s
//! block-64 online softmax are re-associations of the same sum, and reproducing them would
//! pin the oracle to one kernel's tiling. Everything that changes a value rather than its
//! summation order IS reproduced — that is the line drawn here, and `forward.rs` states
//! what it costs in tolerance.

// jscpd:ignore-start
//
// MIRRORING IS THE POINT for the block below, so the duplication gate is off for it. Each
// function is a statement-for-statement transcription of a `kernel.py` primitive, and the
// oracle's whole value is that it was written from the reference rather than from rivoli's
// own idea of what fp8 means. `src/math.rs` and `kernels/common.hpp` contain arithmetic of
// the same SHAPE (an e4m3 encoder, an e8m0 decoder) for GLM's formats; factoring the two
// together would leave this file testing rivoli against itself, which is the "check that
// passes without examining anything" failure this repo has already been bitten by twice
// (docs/reference/architecture.md §8b, tests/glsl_numerics.rs).

/// `float8_e4m3fn` decode. 1-4-3, bias 7, max 448, **no infinities**; `S.1111.111` is NaN.
///
/// Transcribed from the format definition CUDA's `__nv_fp8_e4m3` implements, which is what
/// `T.Cast(FP8, …)` in `act_quant_kernel` lowers to.
pub fn e4m3_decode(b: u8) -> f32 {
    let sign = if (b & 0x80) != 0 { -1.0f32 } else { 1.0 };
    let exp = ((b >> 3) & 0x0f) as i32;
    let mant = (b & 0x07) as f32;
    if exp == 0 {
        // Subnormal: quantum is 2^-9.
        return sign * mant * (1.0 / 512.0);
    }
    if exp == 15 && mant == 7.0 {
        return f32::NAN;
    }
    sign * (1.0 + mant * 0.125) * f32::from_bits(((exp - 7 + 127) as u32) << 23)
}

/// `float8_e4m3fn` encode, **round-to-nearest-even**, saturating at ±448.
///
/// RNE all the way down INCLUDING the subnormal range. That is not the same rule as
/// `kernels/common.hpp::f2e4m3`, which rounds half-away-from-zero below 2^-6 — see
/// `tests/glsl_numerics.rs`. rivoli's rule is rivoli's; V4 was trained against CUDA's
/// `cvt.rn.satfinite.e4m3x2.f32`, which is RNE, so RNE is what the oracle must model. The
/// difference is one ulp on exact halfway subnormals and nothing elsewhere;
/// `tests/v4_oracle.rs::e4m3_encode_is_nearest_ties_to_even` proves this implementation is
/// nearest-ties-even by comparing it against an enumeration of all 254 finite codes.
pub fn e4m3_encode(x: f32) -> u8 {
    if x.is_nan() {
        return 0x7f;
    }
    let sign: u8 = if x.is_sign_negative() { 0x80 } else { 0x00 };
    let a = x.abs();
    // 448 is the largest finite magnitude; 464 is the midpoint to the (absent) next code, so
    // RNE sends anything below it to 448 and the format saturates above.
    if a >= 464.0 {
        return sign | 0x7e;
    }
    // Below half the subnormal quantum (2^-10) everything rounds to zero; exactly 2^-10 is a
    // tie between 0 (mantissa 0, even) and 2^-9 (mantissa 1, odd), so it rounds to zero too.
    if a <= 1.0 / 1024.0 {
        return sign;
    }
    let bits = a.to_bits();
    let e = ((bits >> 23) & 0xff) as i32 - 127;
    if e < -6 {
        // Subnormal: value = m * 2^-9, m in 1..=7 (m == 8 promotes to the smallest normal).
        let scaled = a * 512.0;
        let mut m = scaled.floor();
        let rem = scaled - m;
        if rem > 0.5 || (rem == 0.5 && (m as u32 & 1) != 0) {
            m += 1.0;
        }
        let m = m as u32;
        return if m >= 8 { sign | 0x08 } else { sign | m as u8 };
    }
    let mant = bits & 0x007f_ffff;
    let mut m3 = mant >> 20;
    let rem = mant & 0x000f_ffff;
    let half_ulp = 0x0008_0000;
    if rem > half_ulp || (rem == half_ulp && (m3 & 1) != 0) {
        m3 += 1;
    }
    let mut exp = e + 7;
    if m3 == 8 {
        m3 = 0;
        exp += 1;
    }
    if exp >= 15 && m3 >= 7 {
        return sign | 0x7e;
    }
    sign | ((exp as u8) << 3) | m3 as u8
}

/// `float8_e8m0fnu` decode: a bare power of two, `2^(b-127)`. `0xff` is NaN.
///
/// This is the scale format for BOTH the fp8 attention weights (128x128 blocks) and the fp4
/// expert weights (group-32 along K) — same encoding, different blocking.
pub fn e8m0_decode(b: u8) -> f32 {
    if b == 0xff {
        return f32::NAN;
    }
    if b == 0 {
        // 2^-127 is BELOW f32's smallest normal (2^-126), so the exponent-field shift below
        // would yield 0.0 rather than the value. It is a subnormal with mantissa 2^22.
        // Unreachable for real scales — `act_quant`'s 1e-4 amax floor puts the smallest
        // producible scale at 2^-22 — but a decoder that is silently wrong on one code is
        // exactly the kind of hole this oracle exists to not have.
        return f32::from_bits(1u32 << 22);
    }
    f32::from_bits((b as u32) << 23)
}

/// `float4_e2m1fn` decode of one nibble. 1-2-1, bias 1: {0, .5, 1, 1.5, 2, 3, 4, 6} and
/// their negatives. No infinities, no NaN.
pub fn e2m1_decode(nib: u8) -> f32 {
    const MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let m = MAG[(nib & 0x07) as usize];
    if (nib & 0x08) != 0 { -m } else { m }
}

/// `float4_e2m1fn` encode, round-to-nearest-even, saturating at ±6.
///
/// The eight magnitudes have seven midpoints, so the code IS the number of midpoints the
/// input has passed. Ties go to the even mantissa bit, which for an ascending table means
/// "round up at an odd-indexed midpoint, down at an even-indexed one": 0.25 -> 0,
/// 0.75 -> 1.0, 1.25 -> 1.0, 1.75 -> 2.0, 2.5 -> 2.0, 3.5 -> 4.0, 5.0 -> 4.0.
///
/// Counting rather than searching also makes saturation fall out: a huge input passes all
/// seven and lands on code 7. A nearest-neighbour search does NOT — `1e9 - 6.0 == 1e9` in
/// f32 makes every candidate equidistant, the tie rule keeps the first, and the encoder
/// returns ZERO. That was a real bug here, caught by
/// `tests/v4_oracle.rs::e2m1_encode_is_nearest_ties_to_even`.
pub fn e2m1_encode(x: f32) -> u8 {
    const MID: [f32; 7] = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0];
    let a = x.abs();
    let code = MID
        .iter()
        .enumerate()
        .filter(|&(i, &m)| a > m || (a == m && i % 2 == 1))
        .count();
    (if x.is_sign_negative() { 0x08 } else { 0x00 }) | code as u8
}

/// `bfloat16` decode — exact and total: every bf16 pattern is a representable f32.
pub fn bf16_decode(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// `bfloat16` encode, round-to-nearest-even, inf/NaN passed through verbatim.
pub fn bf16_encode(x: f32) -> u16 {
    let b = x.to_bits();
    if (b & 0x7f80_0000) == 0x7f80_0000 {
        return (b >> 16) as u16;
    }
    ((b + (((b >> 16) & 1) + 0x7fff)) >> 16) as u16
}

/// `kernel.py::fast_log2_ceil` — `ceil(log2(x))` by IEEE-754 bit surgery.
pub fn fast_log2_ceil(x: f32) -> i32 {
    let bits_x = x.to_bits();
    let exp_x = ((bits_x >> 23) & 0xff) as i32;
    let man_bits = bits_x & ((1 << 23) - 1);
    exp_x - 127 + if man_bits != 0 { 1 } else { 0 }
}

/// `kernel.py::fast_pow2` — `2^x` for integer `x` by IEEE-754 bit surgery.
pub fn fast_pow2(x: i32) -> f32 {
    f32::from_bits(((x + 127) as u32) << 23)
}

/// `kernel.py::fast_round_scale` — the ue8m0 scale rule: round the block scale UP to the
/// next power of two so it is exactly representable in `float8_e8m0fnu`.
pub fn fast_round_scale(amax: f32, max_inv: f32) -> f32 {
    fast_pow2(fast_log2_ceil(amax * max_inv))
}

// jscpd:ignore-end

/// FP8_MAX from `act_quant_kernel`. Also the clamp bound.
pub const FP8_MAX: f32 = 448.0;
/// FP4_MAX from `fp4_quant_kernel`.
pub const FP4_MAX: f32 = 6.0;

/// The shape both `act_quant` and `fp4_quant` have: per block along the last dimension,
/// take `amax`, floor it, derive a scale, then round-trip every element through the target
/// format at that scale and write it back.
///
/// Factored rather than written twice because the two kernels differ ONLY in their four
/// constants and their codec — and a copy would let one drift from the other, which for a
/// quantization simulator is a silent-wrong of exactly the kind this file exists to catch.
fn simulate_block_quant(
    row: &mut [f32],
    block: usize,
    amax_floor: f32,
    max: f32,
    round_scale: bool,
    roundtrip: fn(f32) -> f32,
) {
    for chunk in row.chunks_mut(block) {
        let amax = chunk
            .iter()
            .fold(0.0f32, |a, v| a.max(v.abs()))
            .max(amax_floor);
        let s = if round_scale {
            fast_round_scale(amax, 1.0 / max)
        } else {
            amax / max
        };
        for v in chunk.iter_mut() {
            *v = roundtrip((*v / s).clamp(-max, max)) * s;
        }
    }
}

/// `kernel.py::act_quant(x, block_size, scale_fmt, scale_dtype, inplace=True)` — the fused
/// quantize-then-dequantize the reference uses to *simulate* fp8 in a bf16 tensor.
///
/// `round_scale` is `scale_fmt is not None`, which the shipped config makes TRUE
/// (`scale_fmt: "ue8m0"`), so the block scale is always a power of two. The `1e-4` amax
/// floor is `T.max(amax_local[i], 1e-4)`: it keeps an all-zero block from producing a zero
/// scale and a division by zero.
///
/// Blocks run along the LAST dimension; `row` is one flattened leading index, matching the
/// kernel's `x.view(-1, N)`.
pub fn act_quant_inplace(row: &mut [f32], block: usize, round_scale: bool) {
    simulate_block_quant(row, block, 1e-4, FP8_MAX, round_scale, |q| {
        e4m3_decode(e4m3_encode(q))
    });
}

/// `kernel.py::fp4_act_quant(x, block_size, inplace=True)`. The scale is ALWAYS rounded —
/// the fp4 kernel has no `round_scale` switch — and the amax floor is `6 * 2^-126`.
pub fn fp4_act_quant_inplace(row: &mut [f32], block: usize) {
    simulate_block_quant(
        row,
        block,
        FP4_MAX * f32::from_bits(1u32 << 23),
        FP4_MAX,
        true,
        |q| e2m1_decode(e2m1_encode(q)),
    );
}

/// `model.py::rotate_activation` — the randomized-Hadamard spread applied before fp4
/// quantization in the indexer and its compressor, `scale = n^-0.5`.
///
/// **The basis order is natural (Sylvester), CONFIRMED 2026-08-05** against
/// `fast_hadamard_transform`'s own documented contract. It shipped as "INFERRED, and the
/// highest-risk inference in this file", because the package is not vendored with the
/// checkpoint and `inference/requirements.txt` does not pin it — so the order could not be
/// read off the reference, only off the dependency. It was, and this implementation was
/// right.
///
/// **`tests/v4_hadamard_basis.rs` holds the evidence chain, the gate and the measurement.**
/// It pins this function to an explicitly constructed Sylvester matrix bit-for-bit and
/// carries its own negative control. Deliberately not restated here: the numbers belong with
/// the code that produces them, and an earlier version of this comment carried a copy that
/// was already wrong when it was written.
///
/// Why it mattered, which is the part worth having at the call site: sequency-vs-natural
/// ordering permutes the fp4 quantization GROUPS without changing any magnitude. Nothing
/// upstream of `fp4_act_quant_inplace` can see it — every candidate order is orthogonal, so
/// the dot product is unchanged — and everything downstream is a different ranking. The
/// consequence had it been wrong is that the *indexer's* top-k selection shifts, changing
/// which positions are attended; no tolerance on an activation can see that, which is why
/// `forward.rs` emits the indexer's selected indices as a golden of their own and S2
/// compares SETS, not just numbers.
pub fn hadamard_rotate(row: &mut [f32]) {
    let n = row.len();
    debug_assert!(n.is_power_of_two(), "Hadamard needs a power-of-two length");
    let mut h = 1usize;
    while h < n {
        let mut i = 0;
        while i < n {
            for j in i..i + h {
                let a = row[j];
                let b = row[j + h];
                row[j] = a + b;
                row[j + h] = a - b;
            }
            i += h * 2;
        }
        h *= 2;
    }
    let scale = (n as f32).sqrt().recip();
    for v in row.iter_mut() {
        *v *= scale;
    }
}

/// `F.softplus(x)` with PyTorch's DEFAULT `threshold=20`, which the reference relies on
/// without naming: above 20 the function is the identity, not `ln(1+e^x)`.
///
/// It matters because `sqrtsoftplus` scoring feeds this to a `sqrt`, and `ln(1+e^x)` in f32
/// saturates to `+inf` for x above ~88 — the identity branch is what keeps large router
/// logits finite. Dropping the threshold is a plausible transcription slip that is silent
/// until a logit gets large.
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

/// `torch.sigmoid`.
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `F.silu` = `x * sigmoid(x)`.
pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}
