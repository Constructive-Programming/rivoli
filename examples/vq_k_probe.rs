//! Probe B for perf-roadmap item #2 (`VQ_K=2048`): what does halving the codebook cost
//! in RECONSTRUCTION ERROR, measured on real GLM-5.2 expert weights, CPU only.
//!
//! **This is a screen, not a quality measurement.** The engine's quality bar is paired
//! dNLL from `bin/ppl`; relative Frobenius error on a weight matrix is a proxy for it and
//! nothing more. It is here because a full answer costs a requantization of 75 layers, and
//! a proxy that comes back clearly bad is enough to not pay that. A proxy that comes back
//! clearly good is NOT enough to ship — it only earns the requant.
//!
//! What it does, per projection (gate/up/down, which have different weight
//! distributions and therefore separate codebooks):
//!
//!   1. Builds the codebook sample EXACTLY as `bin/convert` does — one routed expert per
//!      MoE layer, `sample_subvectors` at the stride that lands on `--sample-target`.
//!   2. Fits BOTH codebooks from that ONE sample with `learn_codebook_k`, same iteration
//!      cap, same RNG stream. Fitting them by two different procedures would measure the
//!      procedures.
//!   3. Encodes HELD-OUT experts — a different expert index in the same layers, never in
//!      the sample — with the shipping encoder `quant_vq` (amax scale, closed-form group
//!      refit, re-assign), decodes with the shipping decoder `vq_decode_proj`, and reports
//!      relative Frobenius error. Same rows, same weights, both K: the only thing that
//!      differs is the codebook.
//!   4. Prints int4 (`quant_i4`, the OTHER shipping format) on the same rows as an anchor,
//!      because an error ratio means nothing without a rung of known PPL beside it:
//!      int4 = 5.120, int3-vq = 5.275.
//!
//! The `k < VQ_K` codebook is padded to `VQ_K` entries with a far-away filler so
//! `quant_vq`'s `0..VQ_K` scan can never select a padded entry — the trick `bin/convert`
//! and the `quant.rs` unit tests already use. That keeps the encoder itself out of the
//! comparison.
//!
//! Run (no GPU, no artifact, no GPU lock — it only reads the fp8 checkpoint over NFS):
//! ```text
//! cargo run --release --example vq_k_probe -- /swarm/storage/ai/openclaw/glm52-fp8
//! ```
#![allow(clippy::expect_used)]

use rivoli::artifact::format::Safetensors;
use rivoli::artifact::quant::{
    VQ_DIM, VQ_K, dequant_i4, expert_projs, learn_codebook_k, quant_i4, quant_vq,
    sample_subvectors, vq_decode_proj, VqProj,
};

/// GLM-5.2, from the fp8 `config.json`. Hardcoded rather than parsed: this probe is
/// pinned to the one checkpoint the roadmap row is about, and a silent shape mismatch
/// would be caught by `dequant_fp8`'s own shape assertion anyway.
const HIDDEN: usize = 6144;
const INTER: usize = 2048;
const N_EXPERTS: usize = 256;
const DENSE_LAYERS: usize = 3;
const N_LAYERS: usize = 78;
/// `quantization_config.weight_block_size = [128, 128]`.
const FP8_BLOCK: usize = 128;

/// The two rates under test. 4096 = 12-bit index = 3.00 bpw (today); 2048 = 11-bit =
/// 2.75 bpw (the proposal). Both must be ≤ `VQ_K` for the padding trick to work.
const KS: [usize; 2] = [4096, 2048];

struct Args {
    fp8_dir: String,
    sample_target: usize,
    iters: usize,
    /// Rows of each held-out matrix to encode. `quant_vq` is O(rows · i_dim/4 · K · 4)
    /// and single-threaded per call, so this is the run-time dial; it is sharded across
    /// cores below. 256 rows of gate is ~1.6e9 distance evaluations per K.
    eval_rows: usize,
    /// How many held-out (layer, expert) pairs to average over.
    eval_experts: usize,
    /// Also fit ONE K=4096 codebook over gate and up together and score both against it.
    ///
    /// Not part of the K question. It prices the alternative Probe A's `shared_gu_k4096`
    /// arm measures on the kernel side: `moe_gateup_vq` gathers through gate's AND up's
    /// codebooks at once, so sharing them halves that kernel's codebook working set
    /// WITHOUT cutting the rate. The rate stays 3.00 bpw and every number here is
    /// comparable to the K=4096 rows above, so the cost of sharing is the whole answer.
    joint: bool,
}

fn args() -> Args {
    let a: Vec<String> = std::env::args().collect();
    let get = |name: &str, default: usize| -> usize {
        a.iter()
            .position(|x| x == name)
            .and_then(|i| a.get(i + 1))
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    Args {
        fp8_dir: a
            .get(1)
            .filter(|s| !s.starts_with('-'))
            .cloned()
            .unwrap_or_else(|| "/swarm/storage/ai/openclaw/glm52-fp8".into()),
        // 2^20 subvectors is what `convert` fits the shipped codebooks on, so the K=4096
        // arm here is the shipped rate at the shipped sample size — 256 points per
        // centroid. Anything smaller would flatter K=2048 by starving K=4096.
        sample_target: get("--sample-target", 1 << 20),
        iters: get("--iters", 40),
        eval_rows: get("--eval-rows", 256),
        eval_experts: get("--eval-experts", 3),
        joint: a.iter().any(|x| x == "--joint"),
    }
}

/// `‖W − Ŵ‖_F / ‖W‖_F` and the mean per-row relative error. Frobenius is the aggregate
/// the rate-distortion argument is about; the row mean is reported beside it because a
/// projection whose error concentrates in a few rows and one that spreads it evenly have
/// the same Frobenius and very different downstream behaviour (that concentration is
/// exactly what killed per-row int4 scales — see `quant.rs`'s int4 module comment).
fn err(w: &[f32], wh: &[f32], i_dim: usize) -> (f64, f64) {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    let mut rows = 0.0f64;
    let mut nrow = 0usize;
    for (a, b) in w.chunks_exact(i_dim).zip(wh.chunks_exact(i_dim)) {
        let (mut rn, mut rd) = (0.0f64, 0.0f64);
        for (&x, &y) in a.iter().zip(b) {
            rn += f64::from(x - y) * f64::from(x - y);
            rd += f64::from(x) * f64::from(x);
        }
        num += rn;
        den += rd;
        if rd > 0.0 {
            rows += (rn / rd).sqrt();
            nrow += 1;
        }
    }
    ((num / den).sqrt(), rows / nrow.max(1) as f64)
}

/// `quant_vq` over `o_dim` rows, sharded across cores. Row-independent by construction —
/// `quant_vq` touches only `w[o·i_dim..]`, `indices[o·rb..]`, `scales[o·ng..]` — so a
/// shard is bit-identical to the same rows inside one whole-matrix call. Threading it is
/// what makes a 256-row × two-K comparison a minute instead of half an hour.
fn quant_vq_par(w: &[f32], o_dim: usize, i_dim: usize, cb: &[f32]) -> Vec<f32> {
    let threads = std::thread::available_parallelism().map_or(4, |t| t.get());
    let chunk = o_dim.div_ceil(threads);
    let parts: Vec<(usize, Vec<f32>)> = std::thread::scope(|s| {
        let mut hs = Vec::new();
        for t in 0..threads {
            let (lo, hi) = (t * chunk, ((t + 1) * chunk).min(o_dim));
            if lo >= hi {
                break;
            }
            hs.push(s.spawn(move || {
                let rows = hi - lo;
                let seg = &w[lo * i_dim..hi * i_dim];
                let (idx, sc) = quant_vq(seg, rows, i_dim, cb);
                let scb: Vec<u8> = sc.iter().flat_map(|v| v.to_le_bytes()).collect();
                let p = VqProj {
                    indices: &idx,
                    scales: &scb,
                    o_dim: rows,
                    i_dim,
                };
                (lo, vq_decode_proj(&p, cb))
            }));
        }
        hs.into_iter().filter_map(|h| h.join().ok()).collect()
    });
    let mut out = vec![0.0f32; o_dim * i_dim];
    for (lo, part) in parts {
        out[lo * i_dim..lo * i_dim + part.len()].copy_from_slice(&part);
    }
    out
}

/// A `k`-entry codebook widened to the `VQ_K` slots `quant_vq` scans, with the unused
/// entries pushed so far away they can never win a nearest-lookup. `1e30` squared
/// overflows to `inf` in the `‖c‖²` precompute, which is the correct answer here: never
/// nearest, and never a NaN.
fn pad(cb: &[f32]) -> Vec<f32> {
    let mut v = vec![1e30f32; VQ_K * VQ_DIM];
    v[..cb.len()].copy_from_slice(cb);
    v
}

fn main() {
    let a = args();
    let src = Safetensors::open_dir(&a.fp8_dir).expect("open fp8 dir");
    let deq = |base: &str, proj: &str, o: usize, i: usize| -> Vec<f32> {
        src.dequant_fp8(&format!("{base}.{proj}"), o, i, FP8_BLOCK)
            .expect("dequant fp8")
    };
    // `convert`'s sample layers: every MoE layer, one expert each, expert index = layer.
    let layers: Vec<usize> = (DENSE_LAYERS..N_LAYERS).collect();
    let per_expert = INTER * HIDDEN / VQ_DIM;
    let stride = (layers.len() * per_expert / a.sample_target).max(1);
    println!(
        "fp8={}  sample: {} layers x 1 expert, stride {stride}  iters={}  \
         eval: {} rows x {} held-out experts",
        a.fp8_dir,
        layers.len(),
        a.iters,
        a.eval_rows,
        a.eval_experts
    );
    println!("K under test: {KS:?}  (12-bit=3.00bpw vs 11-bit=2.75bpw for the indices)\n");

    // Held-out (layer, expert) pairs, spread across the depth. The sample uses expert `l`
    // of layer `l`, so `l + 128 mod 256` is disjoint from it in every layer.
    let picks: Vec<usize> = (0..a.eval_experts)
        .map(|j| layers[(j + 1) * layers.len() / (a.eval_experts + 1)])
        .collect();

    // `convert`'s sample: one routed expert per MoE layer, appended at `stride`.
    let sample_of = |proj: &str, o_dim: usize, i_dim: usize, stride: usize, out: &mut Vec<f32>| {
        for &l in &layers {
            let base = format!("model.layers.{l}.mlp.experts.{}", l % N_EXPERTS);
            sample_subvectors(&deq(&base, proj, o_dim, i_dim), i_dim, stride, out);
        }
    };
    // Mean (relFrob, rowmean) of `cb` over the held-out experts, on the first `eval_rows`
    // rows of each. Same rows for every codebook, so only the codebook differs.
    let eval_of = |proj: &str, o_dim: usize, i_dim: usize, cb: &[f32]| -> (f64, f64) {
        let (mut f, mut r) = (0.0, 0.0);
        for &l in &picks {
            let base = format!("model.layers.{l}.mlp.experts.{}", (l + N_EXPERTS / 2) % N_EXPERTS);
            let full = deq(&base, proj, o_dim, i_dim);
            let rows = a.eval_rows.min(o_dim);
            let w = &full[..rows * i_dim];
            let (df, dr) = err(w, &quant_vq_par(w, rows, i_dim, cb), i_dim);
            f += df;
            r += dr;
        }
        (f / picks.len() as f64, r / picks.len() as f64)
    };

    for (p, &(proj, (o_dim, i_dim))) in expert_projs(HIDDEN, INTER).iter().enumerate() {
        let t0 = std::time::Instant::now();
        let mut sample = Vec::new();
        sample_of(proj, o_dim, i_dim, stride, &mut sample);
        let nsub = sample.len() / VQ_DIM;
        println!(
            "[{p}] {proj} ({o_dim}x{i_dim})  sample {nsub} subvectors in {:.1}s",
            t0.elapsed().as_secs_f64()
        );
        let cbs: Vec<(usize, Vec<f32>)> = KS
            .iter()
            .map(|&k| {
                let t = std::time::Instant::now();
                let cb = learn_codebook_k(&sample, a.iters, k);
                println!(
                    "     k-means K={k:<5} {:.1}s  ({:.0} points/centroid)",
                    t.elapsed().as_secs_f64(),
                    nsub as f64 / k as f64
                );
                (k, pad(&cb))
            })
            .collect();
        drop(sample);

        let scored: Vec<(usize, (f64, f64))> = cbs
            .iter()
            .map(|(k, cb)| (*k, eval_of(proj, o_dim, i_dim, cb)))
            .collect();
        let base = scored[0].1.0;
        println!("  -> {proj} mean over {} held-out experts:", picks.len());
        for (k, (f, r)) in &scored {
            println!(
                "       K={k:<5} relFrob {f:.5}  rowmean {r:.5}   ({:+.2}% vs K=4096)",
                (f / base - 1.0) * 100.0
            );
        }
        // int4 on the SAME rows. An error ratio is uninterpretable without a rung of known
        // PPL beside it: int4 = 5.120, int3-vq (K=4096) = 5.275, and those two points are
        // what turns "+18.7% relFrob" into a PPL guess at all.
        let (mut fi, mut ri) = (0.0, 0.0);
        for &l in &picks {
            let base_n = format!("model.layers.{l}.mlp.experts.{}", (l + N_EXPERTS / 2) % N_EXPERTS);
            let full = deq(&base_n, proj, o_dim, i_dim);
            let rows = a.eval_rows.min(o_dim);
            let w = &full[..rows * i_dim];
            let (pk, sc) = quant_i4(w, rows, i_dim);
            let (f, r) = err(w, &dequant_i4(&pk, &sc, rows, i_dim), i_dim);
            fi += f / picks.len() as f64;
            ri += r / picks.len() as f64;
        }
        println!(
            "       int4  relFrob {fi:.5}  rowmean {ri:.5}   ({:+.2}% vs K=4096)\n",
            (fi / base - 1.0) * 100.0
        );
    }

    if a.joint {
        // ONE K=4096 codebook for gate AND up. Same rate (3.00 bpw), same 12-bit index,
        // same entry count — the only change is that `moe_gateup_vq`'s two gathers hit one
        // 32 KiB table instead of two, which is the working-set halving Probe A's
        // `shared_gu_k4096` arm times. `stride` doubles so the pooled sample is the same
        // 2^20 subvectors the separate fits got; a bigger sample here would be the joint
        // arm winning on sample size rather than on shareability.
        println!("[joint] one K=4096 codebook shared by gate_proj and up_proj");
        let t0 = std::time::Instant::now();
        let mut sample = Vec::new();
        for &(proj, (o, i)) in &expert_projs(HIDDEN, INTER)[..2] {
            sample_of(proj, o, i, stride * 2, &mut sample);
        }
        println!(
            "     sample {} subvectors in {:.1}s",
            sample.len() / VQ_DIM,
            t0.elapsed().as_secs_f64()
        );
        let t = std::time::Instant::now();
        let cb = pad(&learn_codebook_k(&sample, a.iters, 4096));
        println!("     k-means K=4096 {:.1}s", t.elapsed().as_secs_f64());
        for &(proj, (o, i)) in &expert_projs(HIDDEN, INTER)[..2] {
            let (f, r) = eval_of(proj, o, i, &cb);
            println!("       {proj:<10} shared-K=4096  relFrob {f:.5}  rowmean {r:.5}");
        }
        println!("     compare against the per-projection K=4096 rows above.");
    }
}
