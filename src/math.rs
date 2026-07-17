//! Scalar math primitives for the reference decode path. Correctness first —
//! these are the oracle the HIP kernels are validated against (M2), not the
//! shipped compute. Everything operates on `f32` slices in place where it can.

/// RMSNorm: `y = x / sqrt(mean(x²) + eps) * weight`. In-place into `out`.
pub fn rmsnorm(out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
    debug_assert_eq!(out.len(), x.len());
    debug_assert_eq!(weight.len(), x.len());
    let n = x.len() as f32;
    let ms = x.iter().map(|&v| v * v).sum::<f32>() / n;
    let inv = 1.0 / (ms + eps).sqrt();
    for ((o, &xi), &w) in out.iter_mut().zip(x).zip(weight) {
        *o = xi * inv * w;
    }
}

/// SiLU (a.k.a. swish): `x * sigmoid(x)`.
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Logistic sigmoid.
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
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
/// deterministic tiebreak. `k` is clamped to `scores.len()`.
pub fn topk(scores: &[f32], k: usize) -> Vec<usize> {
    let k = k.min(scores.len());
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    // Partial order would suffice, but n_experts is small (256) and this keeps
    // the tiebreak explicit and deterministic across runs.
    idx.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    idx.truncate(k);
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rmsnorm_unit_weight_normalizes() {
        let x = [3.0f32, 4.0, 0.0, 0.0]; // mean sq = 25/4 = 6.25, rms = 2.5
        let w = [1.0f32; 4];
        let mut out = [0.0f32; 4];
        rmsnorm(&mut out, &x, &w, 0.0);
        // x / 2.5
        assert!((out[0] - 1.2).abs() < 1e-5, "{out:?}");
        assert!((out[1] - 1.6).abs() < 1e-5, "{out:?}");
    }

    #[test]
    fn silu_and_sigmoid_known_points() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
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
