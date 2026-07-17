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
pub fn topk(scores: &[f32], k: usize) -> Vec<usize> {
    let k = k.min(scores.len());
    if k == 0 {
        return Vec::new();
    }
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    // value-desc, index-asc tiebreak — deterministic across runs.
    let cmp = |a: &usize, b: &usize| {
        scores[*b]
            .partial_cmp(&scores[*a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(b))
    };
    if k < idx.len() {
        idx.select_nth_unstable_by(k - 1, cmp);
        idx.truncate(k);
    }
    idx.sort_by(cmp);
    idx
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
}
