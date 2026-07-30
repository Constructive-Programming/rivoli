//! Scalar math primitives for the reference decode path. Correctness first —
//! these are the oracle the HIP kernels are validated against (M2), not the
//! shipped compute. Everything operates on `f32` slices in place where it can.

/// SiLU (a.k.a. swish): `x * sigmoid(x)`.
#[inline]
pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// Logistic sigmoid — the MoE router's scoring function (scoring_func=sigmoid).
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
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

/// Host MoE routing: sigmoid the gate logits into `scores`, add the router `bias` into
/// `choice`, and select the top-`top_k` into `sel`. A free fn over disjoint slices so the
/// GPU engine can borrow `bias` out of `&Pin` while it mutably borrows its own routing
/// scratch. Lives here, not in `gpu.rs`, because it is pure host math and testable without
/// a backend.
///
/// **Routing does not consult the cache, and that is now a load-bearing property.** The
/// `top-m` cache-conditional substitution (arXiv:2412.00099) that used to live here was
/// removed 2026-07-30: it cost +3.63% perplexity on int3-vq and failed outright on int4 at
/// +12.7%, and the LOOKA hint layer supersedes it by steering EVICTION instead of
/// SELECTION. Because selection is now a pure function of (logits, bias, top_k), any cache
/// change — hints, policy, budget — is output-bit-identical by construction, which is the
/// acceptance test for the whole hint mechanism. Re-introducing residency here would
/// silently give that up. See docs/CACHE_ROUTE.md for the retirement record.
pub fn route_into(
    gate_logits: &[u8],
    bias: &[f32],
    top_k: usize,
    scores: &mut [f32],
    choice: &mut [f32],
    sel: &mut Vec<usize>,
) {
    for (s, c) in gate_logits.chunks_exact(4).zip(scores.iter_mut()) {
        *c = sigmoid(f32::from_le_bytes([s[0], s[1], s[2], s[3]]));
    }
    for ((c, &s), &b) in choice.iter_mut().zip(scores.iter()).zip(bias) {
        *c = s + b;
    }
    topk_into(choice, top_k, sel);
}

/// FNV-1a 64 of `bytes`, hex. Identity tag for the `--ppl` corpus, logged beside the
/// numbers so a result can be checked against the text that produced it.
///
// ponytail: not a cryptographic hash and does not need to be — the corpus is committed,
// so git already holds its real identity; this only has to catch "the file changed under
// us". A sha256 would mean taking a dependency to print one line.
pub fn fnv1a64_hex(bytes: &[u8]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h = (h ^ b as u64).wrapping_mul(0x1000_0000_01b3);
    }
    format!("{h:016x}")
}

/// `-log softmax(logits)[target]` from a little-endian f32 logit vector.
///
/// Shifted by the max before exponentiating, which is load-bearing rather than tidy:
/// this model's raw logits routinely run past 80, `exp(80)` overflows f32 to `inf`, and
/// an `inf` in the sum yields a NaN NLL. A NaN would then propagate into a mean that
/// still *looks* like a plausible perplexity, so the failure would be silent and would
/// land in a number we are using to decide a feature. Accumulated in f64 for the same
/// reason — 154,880 f32 addends lose real precision to rounding.
pub fn nll_of(logits_le: &[u8], target: usize) -> anyhow::Result<f32> {
    let n = logits_le.len() / 4;
    anyhow::ensure!(n > 0 && target < n, "target {target} outside {n} logits");
    let z = |i: usize| {
        let b = &logits_le[4 * i..4 * i + 4];
        f32::from_le_bytes([b[0], b[1], b[2], b[3]])
    };
    let mut max = f32::NEG_INFINITY;
    for i in 0..n {
        let v = z(i);
        if v > max {
            max = v;
        }
    }
    anyhow::ensure!(max.is_finite(), "non-finite logits");
    let sum: f64 = (0..n).map(|i| ((z(i) - max) as f64).exp()).sum();
    let nll = (sum.ln() - (z(target) - max) as f64) as f32;
    anyhow::ensure!(nll.is_finite(), "non-finite NLL");
    Ok(nll)
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn bf16_known_patterns_and_rne() {
        // Exact high-16-bit truncation for representable values.
        assert_eq!(f32_to_bf16(0.0), 0x0000);
        assert_eq!(f32_to_bf16(1.0), 0x3f80);
        assert_eq!(f32_to_bf16(-2.0), 0xc000);
        assert_eq!(bf16_to_f32(0x3f80), 1.0);
        assert_eq!(bf16_to_f32(0xc000), -2.0);
        // Round-to-nearest-even at the tie: 1.0 + 2^-8 sits exactly between two
        // bf16 values; the even neighbour is 1.0 (mantissa LSB 0), so it rounds
        // DOWN to 0x3f80 rather than up to 0x3f81.
        assert_eq!(f32_to_bf16(1.0 + 2f32.powi(-8)), 0x3f80);
        // Just past the tie rounds up to the odd neighbour.
        assert_eq!(f32_to_bf16(1.0 + 2f32.powi(-8) + 2f32.powi(-16)), 0x3f81);
        // Non-finite: Inf survives; NaN stays a NaN.
        assert_eq!(f32_to_bf16(f32::INFINITY), 0x7f80);
        assert_eq!(bf16_to_f32(0x7f80), f32::INFINITY);
        assert!(bf16_to_f32(f32_to_bf16(f32::NAN)).is_nan());
    }

    /// `f32_to_f16` is IEEE binary16 round-to-nearest-even. Written against the FORMAT,
    /// not against the hand-rolled RNE it replaced: an implementation-vs-implementation
    /// test proves agreement, which is not the property anyone needs. Every claim here is
    /// one the HIP `__float2half` the codebook is decoded with must also satisfy.
    #[test]
    fn fp16_is_ieee_rne() {
        // 1. EVERY representable fp16 survives a round trip through f32 — 65,536 values,
        //    so this covers all 1,024 subnormals, both zeros and both infinities. A
        //    narrowing bug in any exponent range shows up here as a changed pattern.
        for bits in 0u32..=0xffff {
            let b = bits as u16;
            let v = half::f16::from_bits(b).to_f32();
            if v.is_nan() {
                continue; // NaN payload is not preserved by design; excluded above too
            }
            assert_eq!(f32_to_f16(v), b, "round trip failed for f16 bits {b:#06x}");
        }
        // 2. Ties round to EVEN, in both directions. 1.0 + 2^-11 sits exactly between
        //    1.0 (0x3c00, mantissa LSB 0) and the next fp16 up (0x3c01), so it rounds
        //    DOWN; the tie above 0x3c01 rounds UP to the even 0x3c02.
        assert_eq!(f32_to_f16(1.0 + 2f32.powi(-11)), 0x3c00);
        assert_eq!(f32_to_f16(1.0 + 2f32.powi(-11) + 2f32.powi(-20)), 0x3c01);
        assert_eq!(f32_to_f16(1.0 + 2f32.powi(-10) + 2f32.powi(-11)), 0x3c02);
        // 3. Overflow saturates to inf, and only past the last tie: 65504 is fp16::MAX,
        //    the tie at 65520 rounds up to inf, and just under it rounds back to MAX.
        assert_eq!(f32_to_f16(65504.0), 0x7bff);
        assert_eq!(f32_to_f16(65519.0), 0x7bff);
        assert_eq!(f32_to_f16(65520.0), 0x7c00);
        assert_eq!(f32_to_f16(f32::INFINITY), 0x7c00);
        // 4. Underflow through the subnormals: 2^-24 is the smallest positive subnormal,
        //    2^-25 is its tie and rounds to the even neighbour ZERO, and just above the
        //    tie rounds up to it. Sign is preserved on the way to zero.
        assert_eq!(f32_to_f16(2f32.powi(-24)), 0x0001);
        assert_eq!(f32_to_f16(2f32.powi(-25)), 0x0000);
        assert_eq!(f32_to_f16(2f32.powi(-25) + 2f32.powi(-40)), 0x0001);
        assert_eq!(f32_to_f16(-2f32.powi(-30)), 0x8000);
        // 5. The subnormal/normal seam: 2^-14 is the smallest NORMAL, and the largest
        //    subnormal is one step below it. Off-by-one in the shift lands here.
        assert_eq!(f32_to_f16(2f32.powi(-14)), 0x0400);
        assert_eq!(f32_to_f16(2f32.powi(-14) - 2f32.powi(-24)), 0x03ff);
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

    // --- routing purity (INV-1) -------------------------------------------------------

    /// Deterministic xorshift, local on purpose: `tests/kernel.rs`'s `Lcg` is another
    /// file's helper and is under repair, and a randomized regression test must not
    /// inherit someone else's generator bug.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        /// A logit in [-8, 8) — the router's actual range, wide enough that exact ties
        /// are rare but not impossible (so the index-asc tiebreak still gets exercised).
        fn logit(&mut self) -> f32 {
            (self.next() >> 40) as f32 / 1_048_576.0 - 8.0
        }
    }

    /// One decision's worth of inputs: `n` gate logits as LE bytes, plus a router bias.
    fn gate_and_bias(r: &mut Rng, n: usize) -> (Vec<u8>, Vec<f32>) {
        let mut g = Vec::with_capacity(n * 4);
        for _ in 0..n {
            g.extend_from_slice(&r.logit().to_le_bytes());
        }
        // The load-balancing bias is small next to the sigmoid it shifts.
        (g, (0..n).map(|_| r.logit() * 0.05).collect())
    }

    /// A VERBATIM copy of `route_into` as it stood before `top-m` was ever added
    /// (HEAD 0894d14) — and, now that `top-m` is removed, also what it must compute
    /// forever after.
    fn route_into_pre(
        gate_logits: &[u8],
        bias: &[f32],
        top_k: usize,
        scores: &mut [f32],
        choice: &mut [f32],
        sel: &mut Vec<usize>,
    ) {
        for (s, c) in gate_logits.chunks_exact(4).zip(scores.iter_mut()) {
            *c = sigmoid(f32::from_le_bytes([s[0], s[1], s[2], s[3]]));
        }
        for ((c, &s), &b) in choice.iter_mut().zip(scores.iter()).zip(bias) {
            *c = s + b;
        }
        topk_into(choice, top_k, sel);
    }

    /// **INV-1: routing is a pure function of (gate logits, bias, top_k) — it never
    /// consults the cache.**
    ///
    /// This test outlived the feature it was written for. It began as `top-m`'s regression
    /// guarantee ("`--cache-policy lru|2q|arc` is byte-identical to pre-top-m"); with
    /// `top-m` deleted it now guards the property the LOOKA hint layer depends on. Because
    /// selection cannot see residency, ANY cache change — a hint, a policy swap, a
    /// different budget — is output-bit-identical BY CONSTRUCTION rather than by
    /// measurement. Re-introducing a residency predicate here would silently cost that,
    /// and no output diff would necessarily catch it (top-m's own damage was +3.63%
    /// perplexity, invisible to a token-ID comparison on a short run).
    ///
    /// Property-style over many random score vectors, asserted against the frozen body
    /// above rather than hand-written expectations, so it is a regression test and not a
    /// restatement of the current code.
    #[test]
    fn inv_1_routing_never_consults_the_cache() {
        let mut r = Rng(0x51D0_9E11);
        for n in [8usize, 32, 256] {
            for _ in 0..64 {
                let (g, bias) = gate_and_bias(&mut r, n);
                for &k in &[1usize, 4, 8] {
                    let k = k.min(n);
                    let (mut s1, mut c1, mut sel1) = (vec![0.0; n], vec![0.0; n], Vec::new());
                    let (mut s2, mut c2, mut sel2) = (vec![0.0; n], vec![0.0; n], Vec::new());
                    route_into(&g, &bias, k, &mut s1, &mut c1, &mut sel1);
                    route_into_pre(&g, &bias, k, &mut s2, &mut c2, &mut sel2);
                    assert_eq!(sel1, sel2, "selection drifted from the frozen routing");
                    assert_eq!(s1, s2, "scores drifted");
                    assert_eq!(c1, c2, "choice drifted");
                    assert_eq!(sel1.len(), k, "selection must be exactly top_k");
                    let mut d = sel1.clone();
                    d.sort_unstable();
                    d.dedup();
                    assert_eq!(d.len(), k, "selection must be DISTINCT experts");
                }
            }
        }
    }
}
