//! Scalar math primitives for the reference decode path. Correctness first —
//! these are the oracle the HIP kernels are validated against (M2), not the
//! shipped compute. Everything operates on `f32` slices in place where it can.

/// RMSNorm in place: `v = v / sqrt(mean(v²) + eps) * weight`. The mean is fully
/// computed before any write, so a single mutable slice is sound (callers that
/// need the input preserved copy it into the destination first).
pub fn rmsnorm(v: &mut [f32], weight: &[f32], eps: f32) {
    debug_assert_eq!(weight.len(), v.len());
    let n = v.len() as f32;
    let ms = v.iter().map(|&x| x * x).sum::<f32>() / n;
    let inv = 1.0 / (ms + eps).sqrt();
    for (vi, &w) in v.iter_mut().zip(weight) {
        *vi = *vi * inv * w;
    }
}

/// RMSNorm writing into `dst` from `src`, with the weight read inline from raw
/// little-endian f32 bytes (the mmap tensor) — no `Vec<f32>` decode and no
/// separate copy pass: `dst = src / sqrt(mean(src²)+eps) * weight`, `src`
/// untouched (so the residual survives). Fuses the copy+norm+weight-decode the
/// decode loop otherwise does in three passes.
pub fn rmsnorm_into_bytes(dst: &mut [f32], src: &[f32], weight_bytes: &[u8], eps: f32) {
    debug_assert_eq!(dst.len(), src.len());
    debug_assert_eq!(weight_bytes.len(), src.len() * 4);
    let n = src.len() as f32;
    let ms = src.iter().map(|&x| x * x).sum::<f32>() / n;
    let inv = 1.0 / (ms + eps).sqrt();
    for ((d, &s), wc) in dst.iter_mut().zip(src).zip(weight_bytes.chunks_exact(4)) {
        let w = f32::from_le_bytes([wc[0], wc[1], wc[2], wc[3]]);
        *d = s * inv * w;
    }
}

/// SiLU (a.k.a. swish): `x * sigmoid(x)`.
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Logistic sigmoid — the MoE router's scoring function (scoring_func=sigmoid).
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Truncate f32 → bf16 (round to nearest even). The KV cache stores latents in
/// bf16 — a free 2× on KV bandwidth with no accuracy story (decided 2026-07-18).
#[inline]
pub fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    // Non-finite: keep the top 16 bits verbatim (the RNE carry could turn a NaN
    // into an Inf), so Inf/NaN survive the round-trip as themselves.
    if !x.is_finite() {
        return (bits >> 16) as u16;
    }
    let round = ((bits >> 16) & 1) + 0x7fff;
    ((bits + round) >> 16) as u16
}

/// Widen bf16 → f32 (exact — bf16 is the high 16 bits of f32).
#[inline]
pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
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

/// Block-quantize a latent row (`len` a multiple of [`E4M3_BLOCK`]) into fp8:
/// per 128-element block, scale = amax/448, then each value → e4m3(v/scale).
/// Writes `len` bytes into `data` and `len/128` scales into `scales`.
pub fn quantize_latent_fp8(latent: &[f32], data: &mut [u8], scales: &mut [f32]) {
    debug_assert_eq!(latent.len() % E4M3_BLOCK, 0);
    debug_assert_eq!(data.len(), latent.len());
    debug_assert_eq!(scales.len(), latent.len() / E4M3_BLOCK);
    for (b, blk) in latent.chunks_exact(E4M3_BLOCK).enumerate() {
        let amax = blk.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        // amax==0 → the block is all zeros; any positive scale reproduces them.
        let scale = if amax > 0.0 { amax / E4M3_MAX } else { 1.0 };
        scales[b] = scale;
        // Divide (not multiply-by-reciprocal) to match the device kernel's
        // `latent[i] / scl` exactly — a single correctly-rounded op, so
        // host- and device-quantized bytes agree bit-for-bit.
        for (i, &x) in blk.iter().enumerate() {
            data[b * E4M3_BLOCK + i] = f32_to_e4m3(x / scale);
        }
    }
}

/// Dequantize one fp8 latent element at flat index `i`: `e4m3(data[i]) *
/// scales[i / 128]`. Inverse of [`quantize_latent_fp8`].
#[inline]
pub fn dequant_latent_fp8(byte: u8, scale: f32) -> f32 {
    e4m3_to_f32(byte) * scale
}

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

/// LayerNorm in place: `v = (v - mean) / sqrt(var + eps) * weight + bias`.
/// The DSA indexer's `k_norm` is a true LayerNorm (it ships a bias), unlike
/// every other norm in the model (RMSNorm).
pub fn layernorm(v: &mut [f32], weight: &[f32], bias: &[f32], eps: f32) {
    debug_assert_eq!(weight.len(), v.len());
    debug_assert_eq!(bias.len(), v.len());
    let n = v.len() as f32;
    let mean = v.iter().sum::<f32>() / n;
    let var = v.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n;
    let inv = 1.0 / (var + eps).sqrt();
    for ((vi, &w), &b) in v.iter_mut().zip(weight).zip(bias) {
        *vi = (*vi - mean) * inv * w + b;
    }
}

/// In-place softmax over a slice (numerically stable).
pub fn softmax(v: &mut [f32]) {
    let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in v.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    if sum > 0.0 {
        for x in v.iter_mut() {
            *x /= sum;
        }
    }
}

/// Indices of the `k` largest values, descending by value, with index as a
/// deterministic tiebreak. `k` is clamped to `scores.len()`. Partitions with
/// `select_nth_unstable` (≈O(n)) and sorts only the `k` selected, instead of a
/// full O(n log n) sort of all n — on the per-token MoE path n=256, k=8.
///
/// Test-only: production routes through [`topk_into`] (zero per-call allocation);
/// this Vec-returning form survives only as an ergonomic oracle for unit tests
/// (here and in `moe.rs`).
#[cfg(test)]
pub fn topk(scores: &[f32], k: usize) -> Vec<usize> {
    let mut out = Vec::new();
    topk_into(scores, k, &mut out);
    out
}

/// Like [`topk`] but fills a caller-owned buffer (reused across calls, so the
/// per-token MoE router allocates nothing). `out` doubles as the index
/// workspace: filled with `0..n`, partitioned, truncated to `k`, sorted.
pub fn topk_into(scores: &[f32], k: usize, out: &mut Vec<usize>) {
    let k = k.min(scores.len());
    out.clear();
    if k == 0 {
        return;
    }
    out.extend(0..scores.len());
    // value-desc, index-asc tiebreak — deterministic across runs.
    let cmp = |a: &usize, b: &usize| {
        scores[*b]
            .partial_cmp(&scores[*a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(b))
    };
    if k < out.len() {
        out.select_nth_unstable_by(k - 1, cmp);
        out.truncate(k);
    }
    out.sort_by(cmp);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rmsnorm_unit_weight_normalizes() {
        let w = [1.0f32; 4];
        let mut v = [3.0f32, 4.0, 0.0, 0.0]; // mean sq = 25/4 = 6.25, rms = 2.5
        rmsnorm(&mut v, &w, 0.0);
        // x / 2.5
        assert!((v[0] - 1.2).abs() < 1e-5, "{v:?}");
        assert!((v[1] - 1.6).abs() < 1e-5, "{v:?}");
    }

    #[test]
    fn silu_known_points() {
        assert!(silu(0.0).abs() < 1e-6);
        // silu(x) = x*sigmoid(x); at large x it approaches x.
        assert!((silu(20.0) - 20.0).abs() < 1e-3);
    }

    #[test]
    fn softmax_sums_to_one_and_orders() {
        let mut v = [1.0f32, 2.0, 3.0];
        softmax(&mut v);
        assert!((v.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(v[2] > v[1] && v[1] > v[0]);
    }

    #[test]
    fn topk_picks_largest_with_stable_tiebreak() {
        let s = [0.1f32, 0.9, 0.5, 0.9];
        // two 0.9s at idx 1 and 3 → lower index first.
        assert_eq!(topk(&s, 3), vec![1, 3, 2]);
        assert_eq!(topk(&s, 10), vec![1, 3, 2, 0]);
    }

    /// The MLA attention kernel (attn.hip) replaces the reference two-pass
    /// softmax with a single-pass flash recurrence. The scalar oracle only ever
    /// runs at test-scale context, so this locks — GPU-free, in CI — that the two
    /// forms agree at the 200k regime the kernel actually targets, where the
    /// online rescaling accumulates over the most steps. Mirrors the kernel's
    /// reduction (score → weighted value) with a scalar value per token.
    #[test]
    fn online_softmax_matches_two_pass_at_200k() {
        let n = 200_000usize;
        // Deterministic scores/values (no rand; no Math.random drift).
        let mut st = 0x243f_6a88_85a3_08d3u64;
        let mut next = || {
            st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (st >> 33) as f32 / (1u64 << 31) as f32 - 1.0 // [-1, 1)
        };
        let s: Vec<f32> = (0..n).map(|_| next() * 8.0).collect(); // spread scores
        let v: Vec<f32> = (0..n).map(|_| next()).collect();

        // Two-pass reference: softmax(s) then Σ p_t·v_t.
        let mut p = s.clone();
        softmax(&mut p);
        let two_pass: f32 = p.iter().zip(&v).map(|(&pi, &vi)| pi * vi).sum();

        // Flash online form (as in mla_latent_attend): running max/sum/acc.
        let (mut m, mut l, mut acc) = (f32::NEG_INFINITY, 0.0f32, 0.0f32);
        for (&si, &vi) in s.iter().zip(&v) {
            let m_new = m.max(si);
            let corr = (m - m_new).exp();
            let pi = (si - m_new).exp();
            l = l * corr + pi;
            acc = acc * corr + pi * vi;
            m = m_new;
        }
        let online = acc / l;

        assert!(
            (online - two_pass).abs() <= 1e-4,
            "online={online} two_pass={two_pass} diff={}",
            (online - two_pass).abs()
        );
    }

    #[test]
    fn e4m3_roundtrip_and_known_values() {
        // Exact power-of-two and simple mantissa values round-trip bit-exact.
        for &(x, want) in &[
            (0.0f32, 0.0f32),
            (1.0, 1.0),
            (-2.0, -2.0),
            (448.0, 448.0),   // max normal
            (1.5, 1.5),       // 1 + 4/8
            (0.0625, 0.0625), // 2^-4
        ] {
            let r = e4m3_to_f32(f32_to_e4m3(x));
            assert_eq!(r, want, "{x} -> {r}");
        }
        // Saturation, not inf/NaN, past the max.
        assert_eq!(e4m3_to_f32(f32_to_e4m3(1000.0)), 448.0);
        assert_eq!(e4m3_to_f32(f32_to_e4m3(-1000.0)), -448.0);
        // In-range values land within e4m3's ~2^-3 relative step.
        for &x in &[0.3f32, -1.7, 12.5, 100.0, 0.011, -55.0] {
            let r = e4m3_to_f32(f32_to_e4m3(x));
            let tol = x.abs() * 0.07 + 1e-3; // 3 mantissa bits ≈ 6.25% + subnormal floor
            assert!((r - x).abs() <= tol, "{x} -> {r} (tol {tol})");
        }
        // NaN maps to the e4m3 NaN code and back to NaN.
        assert!(e4m3_to_f32(f32_to_e4m3(f32::NAN)).is_nan());
    }

    #[test]
    fn layernorm_zero_mean_unit_var() {
        // With weight=1 bias=0, output must have ~zero mean and ~unit variance.
        let mut v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 100.0];
        let w = vec![1.0f32; 6];
        let b = vec![0.0f32; 6];
        layernorm(&mut v, &w, &b, 1e-6);
        let mean: f32 = v.iter().sum::<f32>() / 6.0;
        let var: f32 = v.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / 6.0;
        assert!(mean.abs() < 1e-5, "mean {mean}");
        assert!((var - 1.0).abs() < 1e-3, "var {var}");
        // Bias shifts, weight scales: check one element algebraically.
        let mut v2 = vec![2.0f32, 4.0];
        layernorm(&mut v2, &[3.0, 3.0], &[10.0, 10.0], 0.0);
        // mean=3, var=1 → normed = [-1, 1] → *3 + 10 = [7, 13]
        assert!(
            (v2[0] - 7.0).abs() < 1e-4 && (v2[1] - 13.0).abs() < 1e-4,
            "{v2:?}"
        );
    }
}
