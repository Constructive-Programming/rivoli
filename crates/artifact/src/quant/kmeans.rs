//! Codebook learning for the VQ-int3 format: the sampler, the k-means++ seed, the threaded
//! Lloyd iteration, and the ‖c‖² precompute the encoder's argmin searches against.
//!
//! **Split out of `quant.rs` on 2026-08-15 by COHESION, not by size alone.** CodeScene scored
//! the single file 8.54 with Low Cohesion (LCOM4) as the file-level finding — past the
//! file-size cliff, a module that holds two unrelated responsibilities is penalised for
//! holding them, and this was the most disconnected of them. The learner FITS a codebook;
//! everything left in `quant.rs` reads or writes bytes at a fixed on-disk layout. Its only
//! call edges out are `vq::subvec`, `vq::group_amax_bf16` and `worker_threads`, each
//! imported below and each already the single owner of its
//! rule — the first two widened to `pub(super)` for exactly this, because a private copy of
//! either fits the codebook in a space the encoder never visits and nothing about the
//! resulting file looks wrong.
//!
//! **Nothing here was rewritten.** Every body and every comment travelled verbatim from
//! `quant.rs`, because in this repo the comments carry the measurements that chose the
//! constants — a paraphrase would drop the evidence and keep the number. The only edits are
//! intra-doc link paths that now cross a module boundary, and the [`Subvectors`]/[`Weights`]
//! names in the signatures, which are aliases for `[f32]` and change no body and no caller.
//!
//! The pub surface is unchanged: `quant.rs` re-exports [`codebook_norms`],
//! [`learn_codebook`], [`learn_codebook_k`] and [`sample_subvectors`], so
//! `rivoli_artifact::quant::learn_codebook` still resolves for `bin/convert` and
//! `crates/engine/tests/kernel.rs` — no caller outside this crate changed.

use super::vq::{VQ_DIM, VQ_GROUP, VQ_K, group_amax_bf16, subvec};
use super::{Subvectors, Weights, worker_threads};
use rivoli_core::num::bf16_to_f32;

// ── codebook learning ───────────────────────────────────────────────────────
//
// Lives here rather than in the converter because it once had a second consumer, the
// per-layer-codebook study, which had to fit codebooks the SAME way the shipped one was
// fitted or the comparison would measure the fitting procedure instead of the thing it
// was pricing. That study closed negative (docs/investigations/codebook-rotation.md,
// 2026-08-01: 0.09% recovered against a 2% bar) and its binary is gone — recover it from
// tag `archive/vq-study`. `convert` is the only caller now.

/// ‖c‖² per codebook entry — the argmin-VQ precompute, so nearest is
/// `argmin(‖c‖² − 2·x·c)` (half the flops of the full squared distance, same
/// argmin/tie-break). Shared by the CPU encoder and the GPU converter.
pub fn codebook_norms(codebook: &Subvectors) -> Vec<f32> {
    (0..VQ_K)
        .map(|k| subvec(codebook, k).iter().map(|&c| c * c).sum())
        .collect()
}

/// Append every `stride`-th group-normalized subvector of `w` to the codebook sample.
pub fn sample_subvectors(w: &Weights, i_dim: usize, stride: usize, out: &mut Vec<f32>) {
    let mut n = 0usize;
    for grp in w.chunks_exact(i_dim).flat_map(|r| r.chunks_exact(VQ_GROUP)) {
        sample_group(grp, stride, &mut n, out);
    }
}

/// Append every `stride`-th subvector of one group, normalized by that group's bf16 amax —
/// the same normalization [`super::vq::VqEncoder::assign`] searches under, so the codebook is
/// fitted in the space the encoder actually visits. `n` is the running subvector counter
/// across the whole tensor, so the stride does not restart at each group.
fn sample_group(grp: &Weights, stride: usize, n: &mut usize, out: &mut Vec<f32>) {
    let inv = 1.0 / bf16_to_f32(group_amax_bf16(grp));
    for sub in grp.chunks_exact(VQ_DIM) {
        if n.is_multiple_of(stride) {
            out.extend(sub.iter().map(|&v| v * inv));
        }
        *n += 1;
    }
}

/// k-means (k-means++ seed, threaded Lloyd, convergence-stopped) → VQ_K·VQ_DIM.
pub fn learn_codebook(sample: &Subvectors, max_iters: usize) -> Vec<f32> {
    learn_codebook_k(sample, max_iters, VQ_K)
}

/// [`learn_codebook`] at an ARBITRARY entry count → `k·VQ_DIM`.
///
/// Exists so a rate study can fit a `k ≠ VQ_K` codebook the SAME way the shipped one is
/// fitted — same k-means++ seed, same RNG stream, same convergence test. The alternative
/// (a second copy of the learner in the study binary) measures the difference between two
/// fitting procedures and calls it a difference between two rates; that is the trap the
/// per-layer-codebook study named when this learner was moved here rather than left in the
/// converter. `learn_codebook` is this function at `k = VQ_K`, so the shipping fit is
/// bit-identical to what it was — there is no second implementation to drift.
///
/// The returned codebook is NOT directly usable by [`super::quant_vq`] when `k < VQ_K`: that
/// encoder scans `0..VQ_K`. Pad to `VQ_K·VQ_DIM` with a far-away filler (`1e30`) so the
/// unused entries can never win a nearest-lookup — the trick `bin/convert` and the unit
/// tests already use.
pub fn learn_codebook_k(sample: &Subvectors, max_iters: usize, k: usize) -> Vec<f32> {
    let n = sample.len() / VQ_DIM;
    assert!(n >= k, "sample {n} < k {k}");
    let mut c = kmeans_pp_seed(sample, n, k);
    let mut prev = f64::INFINITY;
    for _ in 0..max_iters {
        let a = assign_all(sample, n, &c);
        update_centroids(&mut c, &a);
        if ((prev - a.dist) / a.dist.max(f64::MIN_POSITIVE)).abs() < 1e-4 {
            break;
        }
        prev = a.dist;
    }
    c
}

/// Lloyd's update: each centroid moves to the mean of the subvectors assigned to it. An
/// EMPTY cluster is filtered out rather than divided by zero — it keeps its k-means++ seed,
/// where collapsing it to the origin would make it the nearest entry to everything.
fn update_centroids(c: &mut Subvectors, a: &Assignment) {
    let live = c
        .chunks_exact_mut(VQ_DIM)
        .zip(a.sum.chunks_exact(VQ_DIM))
        .zip(&a.cnt)
        .filter(|&(_, &n)| n > 0);
    for ((cj, sj), &n) in live {
        for (cd, &sd) in cj.iter_mut().zip(sj) {
            *cd = sd / n as f32;
        }
    }
}

/// Squared L2 between one subvector and one centroid — the inner product every phase of
/// the k-means shares. One owner so the seeding's distance and the assignment's distance
/// cannot drift (they were two inline copies before the 2026-08-15 decomposition).
fn sqdist(v: &Subvectors, c: &Subvectors) -> f32 {
    v.iter().zip(c).map(|(&x, &y)| (x - y) * (x - y)).sum()
}

/// k-means++ seeding: each next centroid drawn proportionally to squared distance from
/// the nearest already-chosen one. Deterministic xorshift, so conversion stays
/// byte-stable across runs (the round-trip gate's whole claim).
fn kmeans_pp_seed(sample: &Subvectors, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; k * VQ_DIM];
    let mut next = xorshift64(0x2545_F491_4F6C_DD1D);
    // Uniform 1.0 rather than a distance, so the FIRST draw is uniform over the sample.
    let mut mind = vec![1.0f32; n];
    for j in 0..k {
        let seed = subvec(sample, draw_by_d2(&mind, next()));
        c[j * VQ_DIM..(j + 1) * VQ_DIM].copy_from_slice(seed);
        update_mind(&mut mind, sample, seed, j == 0);
    }
    c
}

/// Deterministic xorshift in `[0, 1)`. Conversion is byte-stable across runs only because
/// this stream is, so the constants are pinned rather than tuned.
fn xorshift64(seed: u64) -> impl FnMut() -> f64 {
    let mut rng = seed;
    move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Draw an index with probability proportional to `d2` — k-means++'s D² rule. `u` is a
/// uniform draw in `[0, 1)`; the last index absorbs the floating-point remainder.
fn draw_by_d2(d2: &[f32], u: f64) -> usize {
    let total: f64 = d2.iter().map(|&d| d as f64).sum();
    let mut t = u * total;
    for (i, &d) in d2.iter().enumerate() {
        t -= d as f64;
        if t <= 0.0 {
            return i;
        }
    }
    d2.len() - 1
}

/// Fold a newly chosen centroid into the "distance to the nearest chosen one" table.
/// `first` REPLACES rather than minimizes: the table starts at a uniform 1.0, which is not a
/// distance and must not be min'd against one.
fn update_mind(mind: &mut [f32], sample: &Subvectors, seed: &Subvectors, first: bool) {
    for (md, v) in mind.iter_mut().zip(sample.chunks_exact(VQ_DIM)) {
        let d = sqdist(v, seed);
        *md = if first { d } else { md.min(d) };
    }
}

/// One assignment pass's result: per-centroid running sums, counts, and the total squared
/// distance. One value because the three are produced together, shipped across the thread
/// boundary together, and merged together — as three loose `Vec`s the merge can pair a sum
/// with the wrong count and still typecheck.
struct Assignment {
    sum: Vec<f32>,
    cnt: Vec<u32>,
    dist: f64,
}

impl Assignment {
    fn zeros(k: usize) -> Self {
        Assignment {
            sum: vec![0.0f32; k * VQ_DIM],
            cnt: vec![0u32; k],
            dist: 0.0,
        }
    }

    /// Add one worker's part. The workers cover disjoint ranges, so this is additive and
    /// the thread count cannot move the result.
    fn merge(&mut self, part: &Assignment) {
        for (a, b) in self.sum.iter_mut().zip(&part.sum) {
            *a += b;
        }
        for (a, b) in self.cnt.iter_mut().zip(&part.cnt) {
            *a += b;
        }
        self.dist += part.dist;
    }
}

/// Index of the centroid nearest `v`, with its squared distance. Kept apart from the
/// seeder's search, which tracks a running minimum over centroids chosen so far rather than
/// scanning the whole set.
fn nearest_centroid(v: &Subvectors, c: &Subvectors) -> (f32, usize) {
    let mut best = (f32::INFINITY, 0usize);
    for (j, cj) in c.chunks_exact(VQ_DIM).enumerate() {
        let d = sqdist(v, cj);
        if d < best.0 {
            best = (d, j);
        }
    }
    best
}

/// One assignment pass over the sample rows in `rows` — the thread body of [`assign_all`].
fn assign_partial(sample: &Subvectors, c: &Subvectors, rows: std::ops::Range<usize>) -> Assignment {
    let mut a = Assignment::zeros(c.len() / VQ_DIM);
    for i in rows {
        let v = subvec(sample, i);
        let (d, j) = nearest_centroid(v, c);
        a.dist += d as f64;
        a.cnt[j] += 1;
        for (s, &x) in a.sum[j * VQ_DIM..(j + 1) * VQ_DIM].iter_mut().zip(v) {
            *s += x;
        }
    }
    a
}

/// The parallel assignment step: fan [`assign_partial`] across threads, merge the parts.
fn assign_all(sample: &Subvectors, n: usize, c: &Subvectors) -> Assignment {
    let threads = worker_threads();
    let chunk = n.div_ceil(threads);
    let parts: Vec<Assignment> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads)
            .map(|t| t * chunk..((t + 1) * chunk).min(n))
            .filter(|r| !r.is_empty())
            .map(|rows| s.spawn(move || assign_partial(sample, c, rows)))
            .collect();
        hs.into_iter().filter_map(|h| h.join().ok()).collect()
    });
    let mut all = Assignment::zeros(c.len() / VQ_DIM);
    for p in &parts {
        all.merge(p);
    }
    all
}
