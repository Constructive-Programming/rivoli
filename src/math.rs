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
pub fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let biased = (bits >> 23) & 0xff;
    let mant = bits & 0x007f_ffff;
    if biased == 0xff {
        // Inf/NaN: keep a mantissa bit set for NaN so it doesn't collapse to Inf.
        return sign | 0x7c00 | if mant != 0 { 0x0200 } else { 0 };
    }
    let exp = biased as i32 - 127 + 15; // rebias 127 → 15
    if exp >= 0x1f {
        return sign | 0x7c00; // overflow → inf
    }
    if exp <= 0 {
        if exp < -10 {
            return sign; // underflow → ±0
        }
        // Subnormal: restore the implicit 1, shift into place with RNE.
        let m = mant | 0x0080_0000;
        let shift = (14 - exp) as u32; // 14..=24
        let half = 1u32 << (shift - 1);
        let rem = m & ((1u32 << shift) - 1);
        let mut out = m >> shift;
        if rem > half || (rem == half && (out & 1) == 1) {
            out += 1;
        }
        return sign | out as u16;
    }
    // Normal: 10-bit mantissa, RNE on the dropped 13 bits.
    let mut e = exp as u32;
    let mut m = mant >> 13;
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (m & 1) == 1) {
        m += 1;
        if m == 0x400 {
            m = 0; // mantissa carry bumps the exponent
            e += 1;
            if e >= 0x1f {
                return sign | 0x7c00;
            }
        }
    }
    sign | ((e as u16) << 10) | m as u16
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
/// `choice`, and select the top-`top_k` into `sel`. A free fn over disjoint slices so
/// the GPU engine can borrow `bias` out of `&Pin` while it mutably borrows its own
/// routing scratch. Lives here, not in `gpu.rs`, because it is pure host math and the
/// `None`-path regression test below has to run without the `rocm` feature.
///
/// `advice = Some((j, m))` switches on **cache-conditional substitution**
/// (`--cache-policy top-m`, arXiv:2412.00099, docs/CACHE_ROUTE.md): the top-`j` ranked
/// candidates are sacred, the remaining `top_k - j` slots prefer candidates that are
/// already RESIDENT and ranked inside the top-`m` window, then fall back to plain rank
/// order. `resident` is asked about expert INDICES (the caller maps them to pool keys)
/// and must not mutate the policy — `HybridPolicy::contains` takes `&self` and does not
/// refresh recency, where `get` would corrupt the eviction clock. `cand` is reusable
/// scratch for the window. Returns the number of chosen slots that were NOT in the true
/// top-`top_k`: the `swap%` numerator.
///
/// Weights are NOT touched: this reorders *selection* only, and the caller still builds
/// its gate values from `scores[e]`.
#[allow(clippy::too_many_arguments)] // disjoint scratch buffers, one call site
pub fn route_into(
    gate_logits: &[u8],
    bias: &[f32],
    top_k: usize,
    advice: Option<(usize, usize)>,
    resident: impl Fn(usize) -> bool,
    scores: &mut [f32],
    choice: &mut [f32],
    sel: &mut Vec<usize>,
    cand: &mut Vec<usize>,
) -> u64 {
    for (s, c) in gate_logits.chunks_exact(4).zip(scores.iter_mut()) {
        *c = sigmoid(f32::from_le_bytes([s[0], s[1], s[2], s[3]]));
    }
    for ((c, &s), &b) in choice.iter_mut().zip(scores.iter()).zip(bias) {
        *c = s + b;
    }
    topk_into(choice, top_k, sel);
    // THE REGRESSION GUARANTEE. lru/2q/arc leave `advice` None, nothing below this line
    // runs, and `sel`/`scores`/`choice` are bit-for-bit what they were before `top-m`
    // existed — checked against a frozen copy of the old body in
    // `advice_none_is_bit_identical_to_the_pre_top_m_routing`.
    let Some((j, m)) = advice else { return 0 };
    // Rank the window off the SAME `choice` with the SAME comparator, so its first
    // `top_k` entries are exactly `sel`. That identity is what lets `cand[..top_k]` stand
    // in for the true top-K when counting swaps, and it is the same invariant bin/replay
    // hard-fails a captured trace over.
    topk_into(choice, m.max(top_k), cand);
    // Transcribed clamp-for-clamp from `substitute` in bin/replay.rs — the simulator that
    // produced the offline (J, M) screen. If the engine and the simulator disagree the
    // screen does not describe this engine, so this is a copy, not a reimplementation.
    let (j, m) = (j.min(top_k), m.min(cand.len()));
    // The true top-K, i.e. what `sel` held a moment ago — `cand`'s prefix IS that
    // ranking, so the swap count below needs no second buffer.
    let true_top = &cand[..top_k.min(cand.len())];
    sel.clear();
    sel.extend(cand.iter().take(j));
    for &e in &cand[j.min(m)..m] {
        if sel.len() == top_k {
            break;
        }
        if resident(e) {
            sel.push(e);
        }
    }
    // Plain rank order for whatever residency could not fill.
    for &e in &cand[j.min(cand.len())..] {
        if sel.len() == top_k {
            break;
        }
        if !sel.contains(&e) {
            sel.push(e);
        }
    }
    sel.iter().filter(|e| !true_top.contains(e)).count() as u64
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
    fn f32_to_f16_known_bit_patterns() {
        // Exact-representable values and a sign.
        assert_eq!(f32_to_f16(0.0), 0x0000);
        assert_eq!(f32_to_f16(1.0), 0x3c00);
        assert_eq!(f32_to_f16(2.0), 0x4000);
        assert_eq!(f32_to_f16(0.5), 0x3800);
        assert_eq!(f32_to_f16(-1.0), 0xbc00);
        // Round to nearest even: 1 + 2^-11 sits exactly between 0x3c00 and 0x3c01,
        // ties to even (0x3c00); 1 + 2^-10 is the next representable step.
        assert_eq!(f32_to_f16(1.0 + 2f32.powi(-11)), 0x3c00);
        assert_eq!(f32_to_f16(1.0 + 2f32.powi(-10)), 0x3c01);
        // Max finite fp16 (65504) and overflow → inf.
        assert_eq!(f32_to_f16(65504.0), 0x7bff);
        assert_eq!(f32_to_f16(1e30), 0x7c00);
        // Underflow → ±0; a NaN keeps a mantissa bit (stays NaN, not inf).
        assert_eq!(f32_to_f16(1e-30), 0x0000);
        assert_ne!(f32_to_f16(f32::NAN) & 0x03ff, 0);
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

    // --- top-m cache-conditional routing (docs/CACHE_ROUTE.md) -----------------------

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

    /// A VERBATIM copy of `route_into` as it stood before `top-m` (HEAD 0894d14). The
    /// test below asserts the shipped function still computes exactly this whenever
    /// `advice` is None. Freezing the old body here — rather than asserting against
    /// hand-written expectations — is what makes it a regression test instead of a
    /// restatement of the new code.
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

    /// THE regression guarantee (docs/CACHE_ROUTE.md "Acceptance": `--cache-policy
    /// lru|2q|arc` byte-identical to today). Property-style over many random score
    /// vectors, and the residency predicate is `unreachable!`, so this also proves the
    /// cache is never even CONSULTED on the None path — a predicate that touched the
    /// policy would be a side effect no output comparison could catch.
    #[test]
    fn advice_none_is_bit_identical_to_the_pre_top_m_routing() {
        const N: usize = 256;
        let mut r = Rng(0x9E37_79B9_7F4A_7C15);
        for top_k in [1usize, 2, 8, 16] {
            for _ in 0..250 {
                let (g, bias) = gate_and_bias(&mut r, N);
                let (mut sa, mut ca, mut la) = (vec![0f32; N], vec![0f32; N], Vec::new());
                let (mut sb, mut cb, mut lb) = (vec![0f32; N], vec![0f32; N], Vec::new());
                let mut cand = Vec::new();
                route_into_pre(&g, &bias, top_k, &mut sa, &mut ca, &mut la);
                let swaps = route_into(
                    &g,
                    &bias,
                    top_k,
                    None,
                    |_| unreachable!("residency must not be queried when advice is None"),
                    &mut sb,
                    &mut cb,
                    &mut lb,
                    &mut cand,
                );
                assert_eq!(swaps, 0, "the None path cannot swap anything");
                assert_eq!(la, lb, "selection diverged at top_k={top_k}");
                // Bit patterns, not float equality: -0.0 == 0.0, and the claim is BYTE
                // identity, not numeric closeness.
                let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&sa), bits(&sb), "scores diverged at top_k={top_k}");
                assert_eq!(bits(&ca), bits(&cb), "choice diverged at top_k={top_k}");
                // `cand` is scratch the None path never writes; it must still be empty.
                assert!(cand.is_empty(), "the None path touched the window scratch");
            }
        }
    }

    /// Drive one substituted decision and hand back `(selection, swaps)`.
    fn sub(
        g: &[u8],
        bias: &[f32],
        top_k: usize,
        j: usize,
        m: usize,
        res: impl Fn(usize) -> bool,
    ) -> (Vec<usize>, u64) {
        let n = bias.len();
        let (mut s, mut c) = (vec![0f32; n], vec![0f32; n]);
        let (mut sel, mut cand) = (Vec::new(), Vec::new());
        let swaps =
            route_into(g, bias, top_k, Some((j, m)), res, &mut s, &mut c, &mut sel, &mut cand);
        (sel, swaps)
    }

    /// M == top_k is a no-op whatever J is — the window IS the selection, so there is
    /// nothing to promote. This is the invariant the offline (J, M) grid rests on: its
    /// M=top_k column is the control that has to reproduce the baseline exactly.
    #[test]
    fn m_equal_to_top_k_is_a_no_op_for_every_j() {
        const N: usize = 256;
        const K: usize = 8;
        let mut r = Rng(0x2545_F491_4F6C_DD1D);
        for _ in 0..200 {
            let (g, bias) = gate_and_bias(&mut r, N);
            let (mut s, mut c) = (vec![0f32; N], vec![0f32; N]);
            let (mut base, mut cand) = (Vec::new(), Vec::new());
            route_into(&g, &bias, K, None, |_| false, &mut s, &mut c, &mut base, &mut cand);
            for j in 0..=K + 2 {
                // Residency is deliberately hostile: EVERY expert resident, so any
                // eligible promotion would fire if the window were wider than top_k.
                let (sel, swaps) = sub(&g, &bias, K, j, K, |_| true);
                assert_eq!(sel, base, "J={j}: M=top_k must reproduce the ranking");
                assert_eq!(swaps, 0, "J={j}: M=top_k cannot swap anything");
            }
        }
    }

    /// The sacred top-J always runs, and residents OUTSIDE the top-M window are never
    /// promoted (the window bound is what stops a resident-but-irrelevant expert from
    /// being routed to). A synthetic descending ranking makes both exact rather than
    /// statistical.
    #[test]
    fn sacred_top_j_survives_and_residents_outside_the_window_do_not_promote() {
        // Descending logits ⇒ expert e ranks e-th. Zero bias keeps the ranking readable.
        const N: usize = 16;
        let mut g = Vec::new();
        for e in 0..N {
            g.extend_from_slice(&(8.0 - e as f32).to_le_bytes());
        }
        let bias = vec![0f32; N];

        // Nothing resident: plain rank order, and the top-J is there because it is
        // sacred, not because it happened to be cached.
        let (sel, swaps) = sub(&g, &bias, 4, 2, 8, |_| false);
        assert_eq!(sel, vec![0, 1, 2, 3]);
        assert_eq!(swaps, 0);

        // Residents at ranks 6/7 are inside M=8: they take the two non-sacred slots, and
        // the sacred prefix 0/1 survives untouched.
        let (sel, swaps) = sub(&g, &bias, 4, 2, 8, |e| e >= 6);
        assert_eq!(sel, vec![0, 1, 6, 7], "top-J kept, the rest swapped to residents");
        assert_eq!(swaps, 2, "ranks 6 and 7 are outside the true top-4");

        // The SAME residents at M=4 are outside the window and must not be promoted.
        let (sel, swaps) = sub(&g, &bias, 4, 1, 4, |e| e >= 6);
        assert_eq!(sel, vec![0, 1, 2, 3], "residents at rank 6/7 are outside M=4");
        assert_eq!(swaps, 0);

        // J=top_k pins the whole selection however the residency falls.
        let (sel, _) = sub(&g, &bias, 4, 4, 16, |e| e >= 8);
        assert_eq!(sel, vec![0, 1, 2, 3], "J=top_k leaves no substitutable slot");
    }

    /// Selection is always exactly `top_k` DISTINCT experts, over the whole (J, M) grid
    /// and every residency pattern — including the degenerate M < J. A short or
    /// duplicated selection would silently change the MoE batch size and the weight
    /// normalization it feeds.
    #[test]
    fn substitution_always_yields_top_k_distinct_experts() {
        const N: usize = 64;
        let mut r = Rng(0xDEAD_BEEF_CAFE_F00D);
        for _ in 0..60 {
            let (g, bias) = gate_and_bias(&mut r, N);
            for top_k in [1usize, 2, 8] {
                for j in 0..=top_k + 1 {
                    for m in [0usize, 1, top_k, top_k + 1, 12, 32, N, N * 4] {
                        for res in 0..4u64 {
                            let (sel, swaps) =
                                sub(&g, &bias, top_k, j, m, |e| (e as u64) % 4 == res);
                            assert_eq!(sel.len(), top_k, "J={j} M={m} res={res}");
                            let uniq: std::collections::HashSet<_> = sel.iter().collect();
                            assert_eq!(uniq.len(), top_k, "duplicate at J={j} M={m} res={res}");
                            // A swap can only land in a non-sacred slot.
                            assert!(
                                swaps as usize <= top_k - j.min(top_k),
                                "swapped a sacred slot at J={j} M={m} res={res}"
                            );
                        }
                    }
                }
            }
        }
    }
}
