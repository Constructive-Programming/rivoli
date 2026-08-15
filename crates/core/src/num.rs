//! Scalar numeric-format conversions — the shared vocabulary quantizers, artifact
//! readers, and host oracles all speak. Ported verbatim from the old tree's math.rs
//! conversion block (`wt/glimmer-s2` @ 6b7f496); the GLM routing/activation oracles
//! that shared that file arrive with M3 in `rivoli-oracles`, not here — an oracle is
//! frozen reference material, a conversion is live vocabulary. (`v4oracle/numerics.rs`
//! keeps its own deliberately separate transliterations, argued in place there.)

#[inline]
pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// Logistic sigmoid — the MoE router's scoring function (scoring_func=sigmoid).
#[inline]
pub(crate) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `sqrt(softplus(x))` — DeepSeek-V4's router affinity, replacing V3's sigmoid.
/// Verified against the reference `inference/model.py`, which computes it as
/// `F.softplus(scores).sqrt()`; guessing this formula would have routed to
/// plausible-but-wrong experts with no crash to notice.
///
/// `softplus` in the stable form `max(x,0) + ln1p(exp(-|x|))`: the naive
/// `ln(1+exp(x))` overflows to `inf` around x=88 in f32 and gate logits are not
/// bounded. Unlike sigmoid this is UNBOUNDED above, so the top-k `choice` values
/// are no longer confined to (0,1) — nothing downstream assumes they are, but the
/// bias is trained against this scale, not sigmoid's.
#[inline]
pub(crate) fn sqrt_softplus(x: f32) -> f32 {
    (x.max(0.0) + (-x.abs()).exp().ln_1p()).sqrt()
}

/// Which affinity the router applies to the gate logits. Part of the pure input
/// tuple INV-1 names, alongside (logits, bias, top_k) — it comes from the model's
/// `scoring_func`, never from residency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Scoring {
    /// GLM-5.2, DeepSeek-V3, Kimi K2.
    #[default]
    Sigmoid,
    /// DeepSeek-V4.
    SqrtSoftplus,
}

impl Scoring {
    #[inline]
    pub fn apply(self, x: f32) -> f32 {
        match self {
            Scoring::Sigmoid => sigmoid(x),
            Scoring::SqrtSoftplus => sqrt_softplus(x),
        }
    }
}

/// Truncate f32 → bf16 (round to nearest even). The KV cache stores latents in
/// bf16 — a free 2× on KV bandwidth with no accuracy story (decided 2026-07-18).
///
/// Delegates to the `half` crate (round-to-nearest-even). Bit-identical to the
/// former hand-rolled RNE for every finite value, zero, subnormal, and infinity;
/// the two differ only on NaN payload bits (never present in the KV/latent/scale
/// data path), where `half` keeps a NaN as a NaN rather than truncating a
/// signaling NaN into an Inf. The kernel's f2bf16 (common.hpp) still matches on
/// this finite domain — the oracle tests never feed NaN.
#[inline]
pub fn f32_to_bf16(x: f32) -> u16 {
    half::bf16::from_f32(x).to_bits()
}

/// Widen bf16 → f32 (exact — bf16 is the high 16 bits of f32). Delegates to
/// `half`; bit-identical to the former `(b as u32) << 16` for every non-NaN
/// pattern.
#[inline]
pub fn bf16_to_f32(b: u16) -> f32 {
    half::bf16::from_bits(b).to_f32()
}

/// Narrow f32 → IEEE-754 binary16 (fp16) bits, round to nearest even. Overflow
/// saturates to ±inf, underflow flushes through the subnormals to ±0. The VQ
/// codebook is stored fp16 (its centroid values sit well inside fp16's normal
/// range, and the per-group bf16 scale carries the magnitude), decoded on the GPU
/// via `__half`; fp16's 10-bit mantissa clears the kernel-oracle tol where bf16's 8
/// does not. Mirror of the device `__float2half`.
///
/// Delegates to `half`, like the two bf16 functions above and for the same reason. This
/// replaced a 45-line hand-rolled RNE with its own subnormal-shift and mantissa-carry
/// paths; `fp16_is_ieee_rne` below pins the contract that made them equivalent — every
/// representable fp16 round-trips, ties round to even, and both boundaries (65504 → inf,
/// 2^-25 → ±0) land where IEEE says — rather than comparing one implementation to the
/// other, which would only prove they agree.
#[inline]
pub fn f32_to_f16(x: f32) -> u16 {
    half::f16::from_f32(x).to_bits()
}

/// Largest finite magnitude representable in OCP `e4m3` (S.1111.110 = 448).
pub const E4M3_MAX: f32 = 448.0;

/// Quantize f32 → OCP `float8_e4m3` (1 sign, 4 exp bias-7, 3 mantissa), round to
/// nearest even, saturating to ±448 (e4m3 has no infinities; the only NaN is
/// 0x7f/0xff). The DeepSeek MLA latent cache stores its NoPE half this way with
/// a per-128 block scale (so inputs here are pre-scaled into e4m3's range). The
/// mirror of [`e4m3_to_f32`].
pub fn f32_to_e4m3(x: f32) -> u8 {
    if x.is_nan() {
        return 0x7f;
    }
    let sign = if x.is_sign_negative() { 0x80u8 } else { 0 };
    let a = x.abs();
    if a >= E4M3_MAX {
        return sign | 0x7e; // saturate to ±448 (S.1111.110)
    }
    // Smallest positive subnormal = 2^-9 (2^-6 · 1/8); below half of it → 0.
    if a < 2f32.powi(-10) {
        return sign;
    }
    let bits = a.to_bits();
    let e = ((bits >> 23) & 0xff) as i32 - 127; // unbiased f32 exponent
    if e < -6 {
        // Subnormal e4m3: value = m/8 · 2^-6. Round a·2^9 to nearest; m==8 means
        // it rounded up to 2^-6 = the smallest NORMAL (exp=1, m3=0), so PROMOTE
        // rather than clamp to 7 — the subnormal analogue of the normal path's
        // m3==8 carry (clamping would return the 2nd-nearest value at that edge).
        let m = (a * 512.0).round() as u8; // 2^9 = 8 · 2^6
        return if m >= 8 { sign | 0x08 } else { sign | m };
    }
    // Normal: exp field e+7 in 1..=15, 3 mantissa bits rounded to nearest even.
    let mant = bits & 0x007f_ffff;
    let mut m3 = (mant >> 20) as u8; // top 3 mantissa bits
    let rem = mant & 0x000f_ffff; // remaining 20 bits
    let half = 0x0008_0000;
    if rem > half || (rem == half && (m3 & 1) == 1) {
        m3 += 1;
    }
    let mut exp = e + 7;
    if m3 == 8 {
        m3 = 0; // mantissa carry → bump exponent
        exp += 1;
    }
    if exp >= 15 && m3 >= 7 {
        return sign | 0x7e; // rounded up into NaN territory → saturate
    }
    sign | ((exp as u8) << 3) | m3
}

/// The MLA fp8 latent cache's block size: one f32 scale per 128 latent values
/// (DeepSeek's 1×128 tile quantization). `kv_lora_rank` must be a multiple.
pub const E4M3_BLOCK: usize = 128;

/// Widen OCP `float8_e4m3` → f32 (exact). Mirror of [`f32_to_e4m3`].
pub fn e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((b >> 3) & 0x0f) as i32;
    let mant = (b & 0x07) as f32;
    if exp == 0 {
        // Subnormal (or zero): value = m/8 · 2^-6.
        sign * (mant / 8.0) * 2f32.powi(-6)
    } else if exp == 15 && mant == 7.0 {
        f32::NAN
    } else {
        sign * (1.0 + mant / 8.0) * 2f32.powi(exp - 7)
    }
}
