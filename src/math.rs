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
}
