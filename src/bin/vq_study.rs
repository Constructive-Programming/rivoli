//! vq_study — how much int3-vq distortion is the RATE, and how much is one codebook
//! being shared by 75 layers?
//!
//! The gating measurement for docs/ROTATION.md. Rotation's remaining argument, and the
//! per-layer-codebook alternative, both claim the same budget:
//!
//! ```text
//!   D_global  =  D_rate       what 12 bits per 4 weights can do at all
//!             +  D_mismatch   what this layer loses by sharing a codebook with 74 others
//! ```
//!
//! Only the sum has ever been observed. `D_mismatch` is what a per-layer codebook recovers
//! at ZERO bytes per expert, and it is the ceiling on what rotation could recover by making
//! every layer's subvector distribution look alike. **If it is small, both die together**
//! and int3-vq is rate-limited — which points at VQ_K, at VQ_GROUP, or at using hybrid.
//!
//! Ground truth is the fp8 checkpoint, so this measures the real quantization error and not
//! a chain of two lossy steps. Everything is CPU + NFS; it never touches the GPU, so it runs
//! beside a decode without taking the device lock.
//!
//! usage: vq_study <fp8-dir> <artifact-dir> [--layers a,b,c] [--experts N] [--rows N]
//!                 [--kmeans-iters N] [--stride N]
//!
//! `<artifact-dir>` supplies the SHIPPED global codebooks (`codebooks.f32`) — the study
//! compares against what actually decodes today, not against a codebook it refits itself.

use anyhow::{Context, Result, bail, ensure};
use rivoli::artifact::format::{Safetensors, load_codebooks};
use rivoli::artifact::quant::{
    VqProj, learn_codebook, quant_vq, sample_subvectors, vq_decode_proj,
    vq_expert_layout, vq_row_bytes,
};

const FP8_BLOCK: usize = 128;
const PROJ: [&str; 3] = ["gate_proj", "up_proj", "down_proj"];

struct Args {
    fp8_dir: String,
    artifact: String,
    layers: Vec<usize>,
    experts: usize,
    rows: usize,
    iters: usize,
    stride: usize,
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        fp8_dir: String::new(),
        artifact: String::new(),
        // Early / middle / late, skipping the 3 dense layers. A spread, not a sweep: at
        // 62 MB/s the cost is expert-count, and six layers is enough to see whether
        // D_mismatch has any layer-dependence at all.
        layers: vec![6, 20, 34, 48, 62, 74],
        // Several per layer, deliberately. One expert would conflate a per-LAYER codebook
        // with a per-EXPERT one, and only per-layer is affordable as header data.
        experts: 8,
        // Rows per projection scored. The error is a mean over rows, so a sample is an
        // unbiased estimate; scoring all 6144 rows would spend minutes in the argmin for
        // a third decimal place that changes no decision.
        rows: 64,
        iters: 25,
        // Subvector stride into the k-means sample.
        stride: 64,
    };
    let mut it = std::env::args().skip(1);
    let mut pos = Vec::new();
    while let Some(t) = it.next() {
        let mut val = |name: &str| -> Result<String> {
            it.next().with_context(|| format!("{name} needs a value"))
        };
        match t.as_str() {
            "--layers" => {
                a.layers = val("--layers")?
                    .split(',')
                    .map(|s| s.trim().parse::<usize>().context("--layers"))
                    .collect::<Result<_>>()?;
            }
            "--experts" => a.experts = val("--experts")?.parse()?,
            "--rows" => a.rows = val("--rows")?.parse()?,
            "--kmeans-iters" => a.iters = val("--kmeans-iters")?.parse()?,
            "--stride" => a.stride = val("--stride")?.parse()?,
            _ => pos.push(t),
        }
    }
    ensure!(
        pos.len() == 2,
        "usage: vq_study <fp8-dir> <artifact-dir> [--layers a,b,c] [--experts N] \
         [--rows N] [--kmeans-iters N] [--stride N]"
    );
    a.fp8_dir = pos[0].clone();
    a.artifact = pos[1].clone();
    Ok(a)
}

/// Relative L2 of a reconstruction against the truth it came from: `‖w−ŵ‖ / ‖w‖`. The
/// scale-free form, so gate (2048×6144) and down (6144×2048) are comparable and so a
/// layer with larger weights does not read as a worse layer.
fn rel_l2(truth: &[f32], recon: &[f32]) -> f64 {
    debug_assert_eq!(truth.len(), recon.len());
    let (mut num, mut den) = (0f64, 0f64);
    for (&t, &r) in truth.iter().zip(recon) {
        let d = (t - r) as f64;
        num += d * d;
        den += (t as f64) * (t as f64);
    }
    if den > 0.0 { (num / den).sqrt() } else { 0.0 }
}

/// Encode `w` with `codebook` and decode it back, then score. The round trip goes through
/// the SHIPPED `quant_vq` / `vq_decode_proj` rather than a reimplementation, so what is
/// measured is the encoder that actually writes `.vq3`.
fn round_trip(w: &[f32], o_dim: usize, i_dim: usize, codebook: &[f32]) -> f64 {
    let (indices, scales) = quant_vq(w, o_dim, i_dim, codebook);
    // `quant_vq` hands back bf16 scales as u16; `VqProj` reads them as LE bytes.
    let sb: Vec<u8> = scales.iter().flat_map(|s| s.to_le_bytes()).collect();
    let p = VqProj { indices: &indices, scales: &sb, o_dim, i_dim };
    debug_assert_eq!(indices.len(), o_dim * vq_row_bytes(i_dim));
    rel_l2(w, &vq_decode_proj(&p, codebook))
}

fn main() -> Result<()> {
    let a = parse_args()?;
    let src = Safetensors::open_dir(&a.fp8_dir)
        .with_context(|| format!("open fp8 checkpoint {}", a.fp8_dir))?;
    let global = load_codebooks(&a.artifact)
        .with_context(|| format!("load shipped codebooks from {}", a.artifact))?;

    let cfg: serde_json::Value =
        serde_json::from_slice(&std::fs::read(format!("{}/config.json", a.fp8_dir))?)?;
    let hidden = cfg["hidden_size"].as_u64().context("hidden_size")? as usize;
    let moe_inter = cfg["moe_intermediate_size"]
        .as_u64()
        .context("moe_intermediate_size")? as usize;
    let n_experts = cfg["n_routed_experts"].as_u64().context("n_routed_experts")? as usize;
    let dense = cfg["first_k_dense_replace"].as_u64().unwrap_or(0) as usize;
    let dims = vq_expert_layout(hidden, moe_inter);
    ensure!(a.experts <= n_experts, "--experts exceeds {n_experts}");
    if let Some(&l) = a.layers.iter().find(|&&l| l < dense) {
        bail!("layer {l} is dense (first {dense}) and has no experts");
    }

    println!(
        "# vq_study — fp8 ground truth, {} experts x {} layers, {} rows/proj scored, \
         k-means {} iters",
        a.experts,
        a.layers.len(),
        a.rows,
        a.iters
    );
    println!("# D_mismatch = refit - per-layer (same fitter, same data volume, pooled vs not).");
    println!(
        "\n{:<6} {:<10} {:>10} {:>10} {:>10} {:>11} {:>9}",
        "layer", "proj", "shipped", "refit", "per-layer", "D_mismatch", "recovered"
    );

    // Phase 1: read once. Every arm below scores the SAME rows and fits from the SAME
    // samples, so the only variable is which data a codebook was trained on.
    struct Cell {
        layer: usize,
        proj: usize,
        sample: Vec<f32>,
        scored: Vec<Vec<f32>>,
    }
    let mut cells: Vec<Cell> = Vec::new();
    for &l in &a.layers {
        // Experts spread across the 256 rather than 0..N: a contiguous prefix would sample
        // one corner of whatever ordering the checkpoint happens to have.
        let ids: Vec<usize> = (0..a.experts)
            .map(|i| (i * n_experts / a.experts) % n_experts)
            .collect();
        for (p, proj) in PROJ.iter().enumerate() {
            let (o_dim, i_dim) = dims[p];
            let row_step = (o_dim / a.rows).max(1);
            let mut sample: Vec<f32> = Vec::new();
            let mut scored: Vec<Vec<f32>> = Vec::with_capacity(ids.len());
            for &e in &ids {
                let w = src.dequant_fp8(
                    &format!("model.layers.{l}.mlp.experts.{e}.{proj}"),
                    o_dim,
                    i_dim,
                    FP8_BLOCK,
                )?;
                sample_subvectors(&w, o_dim, i_dim, a.stride, &mut sample);
                let mut rows = Vec::with_capacity(a.rows * i_dim);
                for r in (0..o_dim).step_by(row_step).take(a.rows) {
                    rows.extend_from_slice(&w[r * i_dim..(r + 1) * i_dim]);
                }
                scored.push(rows);
            }
            eprintln!(
                "vq_study: L{l} {proj}: {} subvectors sampled, {} rows scored",
                sample.len() / 4,
                scored.len() * a.rows
            );
            cells.push(Cell { layer: l, proj: p, sample, scored });
        }
    }

    // THE CONTROL, and the study is worthless without it. A per-layer codebook fitted by
    // this tool could beat or lose to the shipped one for a reason that has nothing to do
    // with per-layer-ness: a different sample size, a different iteration count, a
    // different seed. So fit a THIRD codebook by the identical procedure on the same
    // subvectors POOLED across layers. `refit` vs `per-layer` is then the apples-to-apples
    // comparison — same fitter, same data volume, differing only in whether the training
    // data crossed layer boundaries. `shipped` vs `refit` is the fidelity check on the
    // fitter itself: if they disagree badly, nothing else on the line means anything.
    let mut refit: Vec<Vec<f32>> = Vec::with_capacity(3);
    for p in 0..3 {
        // EQUALIZED. Pooling n layers would otherwise hand the control n times the
        // training subvectors, and k-means gets better with data — so a naive pool would
        // beat per-layer for a reason that has nothing to do with crossing layers, and the
        // study would report "sharing is fine" when it had only measured "more data is
        // better". Take a strided slice so both arms fit the SAME count of subvectors.
        let per_cell = cells
            .iter()
            .filter(|c| c.proj == p)
            .map(|c| c.sample.len())
            .min()
            .unwrap_or(0);
        let n_layers = cells.iter().filter(|c| c.proj == p).count().max(1);
        let mut pooled: Vec<f32> = Vec::with_capacity(per_cell);
        for c in cells.iter().filter(|c| c.proj == p) {
            // every n_layers-th subvector from each layer -> |pooled| ~= |one layer|
            for sub in c.sample.chunks_exact(4).step_by(n_layers) {
                pooled.extend_from_slice(sub);
            }
        }
        eprintln!(
            "vq_study: refit control for {}: {} pooled subvectors (equalized against \
             {} per layer)",
            PROJ[p],
            pooled.len() / 4,
            per_cell / 4
        );
        refit.push(learn_codebook(&pooled, a.iters));
    }

    let mut worst = 0f64;
    let mut all: Vec<f64> = Vec::new();
    let mut fidelity: Vec<f64> = Vec::new();
    for c in &cells {
        let (_, i_dim) = dims[c.proj];
        let per_layer = learn_codebook(&c.sample, a.iters);
        let n = c.scored.len() as f64;
        let (mut ds, mut dr, mut dl) = (0f64, 0f64, 0f64);
        for rows in &c.scored {
            let ro = rows.len() / i_dim;
            ds += round_trip(rows, ro, i_dim, &global[c.proj]);
            dr += round_trip(rows, ro, i_dim, &refit[c.proj]);
            dl += round_trip(rows, ro, i_dim, &per_layer);
        }
        let (ds, dr, dl) = (ds / n, dr / n, dl / n);
        let gap = dr - dl;
        let pct = if dr > 0.0 { 100.0 * gap / dr } else { 0.0 };
        all.push(pct);
        worst = worst.max(pct);
        fidelity.push(if ds > 0.0 { 100.0 * (dr - ds) / ds } else { 0.0 });
        println!(
            "{:<6} {:<10} {ds:>10.6} {dr:>10.6} {dl:>10.6} {gap:>11.6} {pct:>8.2}%",
            c.layer, PROJ[c.proj]
        );
    }

    all.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    let median = all[all.len() / 2];
    let fid = fidelity.iter().sum::<f64>() / fidelity.len() as f64;
    println!(
        "\nmedian recovered {median:.2}%   max {worst:.2}%   (n = {} cells)",
        all.len()
    );
    println!(
        "control: refit-global is {fid:+.2}% vs the SHIPPED global codebook \
         (near 0 = this tool fits like the converter did; large = the numbers above are \
         about the fitter, not about per-layer-ness)"
    );
    if fid.abs() > 10.0 {
        println!(
            "WARNING: the refit control is {fid:+.2}% off the shipped codebook. Raise \
             --kmeans-iters and/or lower --stride until this is near zero BEFORE reading \
             the verdict below."
        );
    }
    // The bar is stated in docs/ROTATION.md and repeated here so a run is self-describing.
    println!(
        "{}",
        if median < 2.0 {
            "VERDICT: BELOW the 2% bar. A per-layer codebook does not pay, and rotation \
             cannot beat it at its own argument — int3-vq is RATE-limited, not \
             codebook-limited. Look at VQ_K / VQ_GROUP / hybrid instead."
        } else {
            "VERDICT: ABOVE the 2% bar. A per-layer codebook recovers real distortion at \
             zero bytes per expert — build that FIRST, then ask whether rotation beats it."
        }
    );
    Ok(())
}
