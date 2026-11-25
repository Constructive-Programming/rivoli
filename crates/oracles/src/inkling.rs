//! Inkling 2's operator references — CPU transliterations of the first-party
//! `transformers.models.inkling2_loop` bodies (transformers @ d8f10a82), the arithmetic the S4
//! GPU kernels twin against. The S3 anchor (`crates/oracles/tests/inkling_anchor_driver.py`) runs
//! the ACTUAL model and vendors its outputs; these are the per-operator references the kernel
//! oracle suites score against, written from the same pinned source.
//!
//! Each function carries the modeling line it transcribes. The fp32 association matches the
//! reference exactly (the S3 tolerance floors are carved fp32-vs-fp64 on these bodies).

/// `Inkling2LoopSoftcap.forward` (MOD:756): `tanh(x/cap)·cap`.
#[inline]
pub fn softcap(x: f32, cap: f32) -> f32 {
    (x / cap).tanh() * cap
}

/// `Inkling2LoopSoftcappedSwiGLU.forward` (MOD:768–771): the sigmoid sees the RAW gate; the
/// softcaps bound only the linear gate factor and `up` (trap T6). fp32 throughout, as the body.
#[inline]
pub fn softcap_swiglu(gate: f32, up: f32, gate_cap: f32, up_cap: f32) -> f32 {
    softcap(gate, gate_cap) * sigmoid(gate) * softcap(up, up_cap)
}

/// `torch.sigmoid` in fp32. The reference uses the ROCm `__ocml` sigmoid on the device; the CPU
/// reference is `1/(1+exp(-x))`, the form the kernel's `exp` intrinsic lowers to.
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `Inkling2LoopL2Clip.forward` (MOD:183–187): scale each head vector down to `threshold` when
/// its L2 norm exceeds it; shorter vectors unchanged. The `+1e-6` and the `clamp(max=1.0)` are
/// both load-bearing (trap T4). `h` is one head vector.
pub fn l2_clip(h: &[f32], threshold: f32) -> Vec<f32> {
    let norm = h.iter().map(|x| x * x).sum::<f32>().sqrt();
    let scale = (threshold / (norm + 1e-6)).min(1.0);
    h.iter().map(|x| x * scale).collect()
}

/// `Inkling2LoopScaledL2Clip.forward` (MOD:147–159): a learned per-head `scale` applied FIRST,
/// then L2Clip. `scale` is one head vector's worth (`head_dim`); the sliding layers' q/k norm.
pub fn scaled_l2_clip(h: &[f32], scale: &[f32], threshold: f32) -> Vec<f32> {
    let scaled: Vec<f32> = h.iter().zip(scale).map(|(x, s)| x * s).collect();
    l2_clip(&scaled, threshold)
}

/// `Inkling2LoopLogScaledL2Clip.forward` (MOD:180–193): own `scale`, L2Clip, then LogScale by `tau`
/// (the full layers' q/k norm). `tau` is `log_scale(1, position, alpha, n_floor)`, precomputed —
/// the position-dependent factor the engine applies per token. Sliding layers omit it (T11).
pub fn log_scaled_l2_clip(h: &[f32], scale: &[f32], threshold: f32, tau: f32) -> Vec<f32> {
    scaled_l2_clip(h, scale, threshold)
        .iter()
        .map(|x| x * tau)
        .collect()
}

/// `Inkling2LoopLogScale.forward` (MOD:215–220): `tau = 1 + alpha·log(max((pos+1)/n_floor, 1))`.
/// `n_floor` is the CONFIG value (8192 shipped), not the class default 128000 (trap T9).
#[inline]
pub fn log_scale(x: f32, position: usize, alpha: f32, n_floor: f32) -> f32 {
    let tau = 1.0 + alpha * (((position + 1) as f32 / n_floor).max(1.0)).ln();
    x * tau
}

/// `Inkling2LoopRMSNorm.forward` (MOD:740–745): T5-style — fp32 variance, `rsqrt(var+eps)`, then
/// `weight·hidden` (weight applied AFTER the rsqrt). `x` is one row.
pub fn rmsnorm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let var = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = (var + eps).sqrt().recip();
    x.iter().zip(weight).map(|(v, w)| w * (v * inv)).collect()
}

/// `Inkling2ShortConvolution.forward` (MOD:349) decode step, state included: a depthwise causal
/// k-tap cross-correlation of the window `[past k−1 tokens, current]` (MOD:255), plus the residual
/// (the current token), divided by `√(k+1)` — T15: the static variant's `/√5` for k=4 (the dynamic
/// variant uses `/√(2k+1)`). `weight` is `[channels, k]`, `state` the running `[channels, k−1]`
/// history, `current` `[channels]`; the geometry derives from the lengths. Returns the output and
/// the ADVANCED state (shift left one, append current — MOD:254), so the caller keeps no separate
/// window copy. The dot accumulates in tap order.
pub fn sconv(weight: &[f32], state: &[f32], current: &[f32], divisor: f32) -> (Vec<f32>, Vec<f32>) {
    let channels = current.len();
    let k = weight.len() / channels;
    sconv_pass((current, state), (k, divisor), |c, cur, st, hist| {
        let (wb, sb) = (c * k, c * hist);
        let mut acc = 0.0f32;
        for j in 0..hist {
            acc += weight[wb + j] * st[sb + j];
        }
        acc + weight[wb + hist] * cur
    })
}

/// The state advance both short-convolution variants share: shift channel `sb`'s `hist`-long
/// history left one and append `cur` (MOD:254). `(sb, hist)` travel as one pair so the helper
/// stays at four arguments.
fn shift_state(state: &[f32], (sb, hist): (usize, usize), cur: f32, next: &mut Vec<f32>) {
    if hist > 0 {
        next.extend_from_slice(&state[sb + 1..sb + hist]);
        next.push(cur);
    }
}

/// The per-channel decode pass both short-convolution variants share: it walks `current`, asks
/// `taps` for that channel's accumulated k-tap dot, writes the dot plus the residual over
/// `divisor`, and advances the channel's history through [`shift_state`] (MOD:254).
///
/// `taps` receives `(channel, current, state, hist)` and returns the dot in its OWN association
/// order, so neither variant's arithmetic — nor its summation order — moves in here: the static
/// variant folds from zero in tap order and adds the current tap last, the dynamic one starts from
/// its modulated current tap. That is also why the two weight layouts never reach this function.
fn sconv_pass(
    (current, state): (&[f32], &[f32]),
    (k, divisor): (usize, f32),
    taps: impl Fn(usize, f32, &[f32], usize) -> f32,
) -> (Vec<f32>, Vec<f32>) {
    let hist = k - 1;
    let mut out = Vec::with_capacity(current.len());
    let mut next = Vec::with_capacity(current.len() * hist);
    for (c, &cur) in current.iter().enumerate() {
        let sb = c * hist;
        out.push((taps(c, cur, state, hist) + cur) / divisor);
        shift_state(state, (sb, hist), cur, &mut next);
    }
    (out, next)
}

/// The dynamic short-convolution's operands, as its kernel takes them: a SHARED static filter
/// `[k, channels]` row-major modulated by a per-head dynamic tap vector `[heads, k]`, the
/// chronological `[channels, k−1]` state, and the current token. Named fields rather than seven
/// positional arguments, where `k`/`head_dim` transposed is a wrong answer, not a type error.
pub struct DynamicSconv<'a> {
    pub static_w: &'a [f32],
    pub dyn_w: &'a [f32],
    pub state: &'a [f32],
    pub current: &'a [f32],
    pub head_dim: usize,
    pub k: usize,
    pub divisor: f32,
}

/// `Inkling2ShortConvolution.forward` (MOD:349) DYNAMIC decode step, otherwise the [`sconv`]
/// body: `w_lag(c) = static_w[lag][c] + dyn_w[c/head_dim][lag]`, lag 0 the current token and lag
/// `j` the state slot `hist − j`. The divisor is the caller's (T15: `√(2k+1) = 3` at k=4). Returns
/// the output and the ADVANCED state, same shift+append as the static variant.
pub fn sconv_dynamic(d: &DynamicSconv<'_>) -> (Vec<f32>, Vec<f32>) {
    let channels = d.current.len();
    sconv_pass(
        (d.current, d.state),
        (d.k, d.divisor),
        |c, cur, st, hist| {
            let head = c / d.head_dim;
            let mut acc = (d.static_w[c] + d.dyn_w[head * d.k]) * cur;
            for lag in 1..d.k {
                let w = d.static_w[lag * channels + c] + d.dyn_w[head * d.k + lag];
                acc += w * st[c * hist + (hist - lag)];
            }
            acc
        },
    )
}

/// The vector update inkling's decode folds into a scaled accumulation: `x[i] + alpha·y[i]`.
pub fn vaxpy(x: &[f32], y: &[f32], alpha: f32) -> Vec<f32> {
    x.iter().zip(y).map(|(&a, &b)| a + alpha * b).collect()
}

/// `eager_attention_forward` (MOD:530–543) single-query decode step with an attention sink: the
/// softmax joins ONE extra sink logit then DROPS its column (T10) — the sink carries zero value
/// but absorbs probability mass, so the real weights sum to < 1. `q` is `[d]`, `k`/`v` are
/// `[n, d]` row-major, `bias` is `[n]` (rel-bias + mask already folded); `n`/`d` derive from the
/// lengths. `logits[j] = (q·k_j)·scaling + bias[j]`; softmax over `[logits, sink]`, max-subtracted.
pub fn sink_attend(qkvb: (&[f32], &[f32], &[f32], &[f32]), sink: f32, scaling: f32) -> Vec<f32> {
    let (q, k, v, bias) = qkvb;
    let d = q.len();
    let n = bias.len();
    let logits: Vec<f32> = (0..n)
        .map(|j| {
            let mut dot = 0.0f32;
            for i in 0..d {
                dot += q[i] * k[j * d + i];
            }
            dot * scaling + bias[j]
        })
        .collect();
    let mut m = sink;
    for &l in &logits {
        m = m.max(l);
    }
    let mut z = (sink - m).exp();
    for &l in &logits {
        z += (l - m).exp();
    }
    let weights: Vec<f32> = logits.iter().map(|&l| (l - m).exp() / z).collect();
    (0..d)
        .map(|i| {
            let mut acc = 0.0f32;
            for j in 0..n {
                acc += weights[j] * v[j * d + i];
            }
            acc
        })
        .collect()
}

/// `Inkling2LoopMessageGate.forward` (MOD:196–200) pointwise tail: `forget = tanh(forget_proj)`,
/// `gain = tanh(message_proj)`, then `(1−forget)·hidden + gain·message` in fp32. The down/forget/
/// gain projections are GEMVs the engine runs separately; this is the gate's combine. The four
/// vectors are one hidden width's worth.
pub fn message_gate(
    forget_logit: &[f32],
    gain_logit: &[f32],
    hidden: &[f32],
    message: &[f32],
) -> Vec<f32> {
    forget_logit
        .iter()
        .zip(gain_logit)
        .zip(hidden)
        .zip(message)
        .map(|(((f, g), h), m)| (1.0 - f.tanh()) * h + g.tanh() * m)
        .collect()
}

/// The KV-cache append scatter: one token's projected `[kv_heads·head_dim]` vector deinterleaved
/// into the per-head cache `[kv_heads, ctx, head_dim]` at position `pos`. Element `e` (head
/// `e/head_dim`, lane `e%head_dim`) lands at `head·ctx·head_dim + pos·head_dim + lane`, so each
/// head's vectors are contiguous for `sink_attend` to read without a gather. A pure data movement —
/// the oracle comparison is bitwise. The returned buffer is zero-filled except the scattered token.
pub fn kv_scatter(src: &[f32], head_dim: usize, ctx: usize, pos: usize) -> Vec<f32> {
    let kv_heads = src.len() / head_dim;
    let mut dst = vec![0.0f32; kv_heads * ctx * head_dim];
    for (e, &v) in src.iter().enumerate() {
        let (head, lane) = (e / head_dim, e % head_dim);
        dst[head * ctx * head_dim + pos * head_dim + lane] = v;
    }
    dst
}

/// The rel-bias distance gather (MOD:224–227) for one decode query at `pos` over the key window
/// `window = (start, n)`: key slot `t` sits at absolute position `start + t`, and `bias[h][t]` is
/// `rel_logits[h][pos−(start+t)]` when `0 <= pos−(start+t) < rel_extent`, else 0 (the masked-out
/// distances). `start > 0` is the sliding window past its first position — reading slot `t`
/// instead of `start + t` would silently use the wrong distance for every key. The `rel_logits`
/// are `[heads, rel_extent]` row-major; the output `[heads, n]`. A pure gather + mask — the
/// oracle comparison is bitwise.
pub fn rel_bias_gather(
    rel_logits: &[f32],
    rel_extent: usize,
    pos: usize,
    window: (usize, usize),
) -> Vec<f32> {
    let (start, n) = window;
    let heads = rel_logits.len() / rel_extent;
    let mut bias = vec![0.0f32; heads * n];
    for (h, row) in rel_logits.chunks_exact(rel_extent).enumerate() {
        for (t, slot) in bias[h * n..(h + 1) * n].iter_mut().enumerate() {
            let key_pos = start + t;
            if pos >= key_pos && pos - key_pos < rel_extent {
                *slot = row[pos - key_pos];
            }
        }
    }
    bias
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softcap_bounds_and_is_odd() {
        assert_eq!(softcap(0.0, 4.0), 0.0);
        // tanh(1)·4 ≈ 3.0464; far past the cap it saturates at the cap.
        assert!((softcap(4.0, 4.0) - 4.0 * 1.0f32.tanh()).abs() < 1e-6);
        assert!((softcap(1e6, 4.0) - 4.0).abs() < 1e-3);
        assert_eq!(softcap(-2.0, 4.0), -softcap(2.0, 4.0));
    }

    #[test]
    fn swiglu_sigmoid_sees_the_raw_gate() {
        // T6: with a tiny gate cap the capped gate factor is ~cap, but the sigmoid still sees the
        // RAW (large) gate, so it is ~1 — sigmoiding the capped gate would give sigmoid(cap) < 1.
        let g = 100.0;
        let out = softcap_swiglu(g, 1.0, 0.01, 25.0);
        let expect = softcap(g, 0.01) * sigmoid(g) * softcap(1.0, 25.0);
        assert_eq!(out, expect);
        assert!(sigmoid(g) > 0.99); // raw gate, not the capped ~0.01
        assert!(out > softcap(g, 0.01) * sigmoid(0.01) * softcap(1.0, 25.0));
    }

    #[test]
    fn softcap_swiglu_is_situ_glu() {
        // S4 reuse, MEASURED 2026-11-24: inkling's softcapped SwiGLU is K3's SiTU-GLU (the
        // `situ_glu_f32` kernel) — bit-identical when both use the division-form sigmoid. The
        // only gap from the torch reference is `torch.sigmoid`'s 1-ulp (1.19e-7) implementation
        // choice, amplified by the gate cap to ~1.9e-6 at the activation output. So the MoE
        // activation needs no new kernel; it reuses `situ_glu_f32` at b1=4, b2=25.
        for (g, u) in [
            (0.5, -0.3),
            (4.0, 25.0),
            (-12.0, 7.0),
            (100.0, -100.0),
            (0.0, 0.0),
        ] {
            let (gc, uc) = (4.0f32, 25.0f32);
            let situ = (gc * (g / gc).tanh()) * sigmoid(g) * (uc * (u / uc).tanh());
            assert_eq!(softcap_swiglu(g, u, gc, uc), situ, "g={g} u={u}");
        }
    }

    #[test]
    fn l2_clip_only_scales_down_and_keeps_epsilon() {
        // A vector longer than threshold is scaled to exactly threshold norm.
        let h = [3.0, 4.0]; // norm 5
        let c = l2_clip(&h, 2.5);
        let n = c.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((n - 2.5).abs() < 1e-5);
        // A short vector is unchanged (clamp max=1).
        let short = [0.3, 0.4]; // norm 0.5 < 2.5
        assert_eq!(l2_clip(&short, 2.5), short.to_vec());
        // The zero vector does not divide by zero (the +1e-6).
        assert_eq!(l2_clip(&[0.0, 0.0], 2.5), vec![0.0, 0.0]);
    }

    #[test]
    fn scaled_l2_clip_applies_the_learned_scale_first() {
        let h = [1.0, 2.0, 3.0];
        let scale = [2.0, 2.0, 2.0];
        // The clip sees the SCALED vector [2,4,6], not [1,2,3].
        assert_eq!(
            scaled_l2_clip(&h, &scale, 3.0),
            l2_clip(&[2.0, 4.0, 6.0], 3.0)
        );
        // A unit scale reduces to the plain clip.
        assert_eq!(scaled_l2_clip(&h, &[1.0, 1.0, 1.0], 8.0), l2_clip(&h, 8.0));
    }

    #[test]
    fn log_scaled_l2_clip_applies_tau_after_the_clip() {
        let h = [1.0, 2.0, 3.0];
        let scale = [2.0, 2.0, 2.0];
        let tau = log_scale(1.0, 16383, 0.1, 8192.0); // > 1 past the floor
        let want: Vec<f32> = scaled_l2_clip(&h, &scale, 3.0)
            .iter()
            .map(|x| x * tau)
            .collect();
        assert_eq!(log_scaled_l2_clip(&h, &scale, 3.0, tau), want);
        // tau == 1 (below the floor) reduces to the scaled clip.
        assert_eq!(
            log_scaled_l2_clip(&h, &scale, 3.0, 1.0),
            scaled_l2_clip(&h, &scale, 3.0)
        );
    }

    #[test]
    fn log_scale_uses_the_config_floor() {
        // Below the floor, tau = 1 (identity). At pos+1 = n_floor, tau = 1.
        assert_eq!(log_scale(2.0, 0, 0.1, 8192.0), 2.0);
        assert_eq!(log_scale(2.0, 8191, 0.1, 8192.0), 2.0);
        // Past the floor it grows; the class default 128000 would still be identity here (T9).
        let shipped = log_scale(2.0, 16383, 0.1, 8192.0);
        let default = log_scale(2.0, 16383, 0.1, 128000.0);
        assert!(shipped > 2.0);
        assert_eq!(default, 2.0);
        assert!(shipped > default);
    }

    #[test]
    fn rmsnorm_applies_weight_after_rsqrt() {
        let x = [1.0, 2.0, 3.0];
        let w = [2.0, 2.0, 2.0];
        let out = rmsnorm(&x, &w, 1e-6);
        let var: f32 = (1.0 + 4.0 + 9.0) / 3.0;
        let inv = 1.0 / (var + 1e-6).sqrt();
        assert!((out[0] - 2.0 * (1.0 * inv)).abs() < 1e-6);
        assert!((out[2] - 2.0 * (3.0 * inv)).abs() < 1e-6);
    }

    #[test]
    fn sconv_static_divisor_is_sqrt_k_plus_1() {
        // T15: k=4 → /√5. Unit weight on the current tap (3), zero past; residual = current.
        let weight = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let state = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let current = vec![2.0, 3.0];
        let (out, next) = sconv(&weight, &state, &current, 5.0f32.sqrt());
        // conv = current (2, 3); (conv + current)/√5 = (4/√5, 6/√5).
        assert!((out[0] - 4.0 / 5.0f32.sqrt()).abs() < 1e-6);
        assert!((out[1] - 6.0 / 5.0f32.sqrt()).abs() < 1e-6);
        // The state shifts left one and appends current: [1,2,3]→[2,3,2], [4,5,6]→[5,6,3].
        assert_eq!(next, vec![2.0, 3.0, 2.0, 5.0, 6.0, 3.0]);
    }

    #[test]
    fn sink_attend_sink_absorbs_probability_mass() {
        // T10: one key, logit 1, sink logit 0. softmax([1,0]) drops the sink → weight 1/(1+e^-1).
        let q = [1.0];
        let k = [1.0];
        let v = [1.0];
        let bias = [0.0];
        let out = sink_attend((&q[..], &k[..], &v[..], &bias[..]), 0.0, 1.0);
        let want = 1.0 / (1.0 + (-1.0f32).exp());
        assert!((out[0] - want).abs() < 1e-6);
        assert!(out[0] < 1.0); // the sink absorbed mass; a sink-less softmax would give 1.0
    }

    #[test]
    fn message_gate_forget_and_gain_gate_the_two_streams() {
        // forget_logit large → forget≈1 → hidden suppressed; gain_logit large → gain≈1 → message kept.
        let f = [10.0];
        let g = [10.0];
        let h = [5.0];
        let m = [7.0];
        let out = message_gate(&f, &g, &h, &m);
        let want = (1.0 - 10.0f32.tanh()) * 5.0 + 10.0f32.tanh() * 7.0;
        assert!((out[0] - want).abs() < 1e-6);
        assert!(out[0] > 6.9); // gain≈1 keeps message≈7; forget≈1 drops hidden
    }

    #[test]
    fn kv_scatter_deinterleaves_into_per_head_blocks() {
        // 2 heads, head_dim 2, ctx 3, pos 1: head h's pair lands at h·6 + 2, contiguous per head.
        let src = [1.0, 2.0, 3.0, 4.0];
        let dst = kv_scatter(&src, 2, 3, 1);
        assert_eq!(
            dst,
            vec![0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0]
        );
    }

    #[test]
    fn rel_bias_gather_reverses_distance_and_masks_out_of_range() {
        // 2 heads, rel_extent 4, pos 2, n 3: bias[h][t] = rel_logits[h][2−t], the reversed first 3.
        let rel_logits = [10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0, 23.0];
        let bias = rel_bias_gather(&rel_logits, 4, 2, (0, 3));
        assert_eq!(bias, vec![12.0, 11.0, 10.0, 22.0, 21.0, 20.0]);
        // pos 1, n 3, rel_extent 2: t=2 has dist −1 (masked to 0); dist 1,0 read indices 1,0.
        let masked = rel_bias_gather(&[10.0, 11.0, 20.0, 21.0], 2, 1, (0, 3));
        assert_eq!(masked, vec![11.0, 10.0, 0.0, 21.0, 20.0, 0.0]);
    }

    #[test]
    fn rel_bias_gather_reads_the_window_start_not_the_slot_index() {
        // A sliding window past its start: pos 10, window (8, 3), rel_extent 4. Slot t is
        // absolute key 8+t, so the distances are 2, 1, 0 → row[2], row[1], row[0]. Reading the
        // slot as the position would give row[10−t], out of range for every t and all zeros.
        let row: Vec<f32> = (0..4).map(|i| i as f32).collect();
        let bias = rel_bias_gather(&row, 4, 10, (8, 3));
        assert_eq!(bias, vec![2.0, 1.0, 0.0]);
        // A window that starts past the query masks everything (no key is at or before `pos`).
        assert_eq!(rel_bias_gather(&row, 4, 10, (11, 2)), vec![0.0, 0.0]);
    }

    #[test]
    fn sconv_dynamic_adds_the_per_head_taps_and_shifts_state() {
        // 2 heads of head_dim 2, k = 2, T15's dynamic divisor /3. Static lag 0 is the current
        // token's tap, lag 1 the previous one; each head's dynamic vector modulates both.
        let static_w = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let dyn_w = [0.5, 0.25, 1.5, 0.75];
        let state = [10.0, 20.0, 30.0, 40.0];
        let current = [2.0, 4.0, 6.0, 8.0];
        let d = DynamicSconv {
            static_w: &static_w,
            dyn_w: &dyn_w,
            state: &state,
            current: &current,
            head_dim: 2,
            k: 2,
            divisor: 3.0,
        };
        let (out, next) = sconv_dynamic(&d);
        // c=0: (1+0.5)·2 + (5+0.25)·10 = 55.5 → (55.5+2)/3; c=1: 10+125 → 139/3.
        // c=2 (head 1): 27+232.5 → 265.5/3; c=3: 44+350 → 402/3.
        let want = [57.5 / 3.0, 139.0 / 3.0, 265.5 / 3.0, 402.0 / 3.0];
        for (i, (want, got)) in want.iter().zip(&out).enumerate() {
            assert!((got - want).abs() < 1e-5, "c={i}: {want} vs {got}");
        }
        // The lag order is load-bearing: swapping lag 0 and lag 1 would give
        // ((5+0.25)·2 + (1+0.5)·10 + 2)/3, which is not this answer.
        assert!((out[0] - 27.5 / 3.0).abs() > 1e-3);
        // Hist 1: the state shifts left and appends the current token.
        assert_eq!(next, current.to_vec());
    }

    #[test]
    fn vaxpy_accumulates_the_scaled_second_vector() {
        let x = [1.0, -2.0, 0.0];
        let y = [2.0, 3.0, -4.0];
        assert_eq!(vaxpy(&x, &y, -2.5), vec![-4.0, -9.5, 10.0]);
        // alpha = 0 is the identity, not a refusal: the update still writes x back unchanged.
        assert_eq!(vaxpy(&x, &y, 0.0), x.to_vec());
    }
}
