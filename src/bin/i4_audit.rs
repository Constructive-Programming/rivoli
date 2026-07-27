//! i4_audit — scale-SENSITIVE audit of the `.i4` expert path against independent
//! ground truth (the original fp8 checkpoint), on real weights.
//!
//! The existing verification compares our GPU kernel to our own `matvec_i4` under
//! cosine, so it is blind to (a) any convention both share and (b) any pure gain
//! error. This tool closes both gaps on the CPU side:
//!
//!   1. on-disk `.i4` bytes  ==  quant_i4(dequant_fp8(fp8 ckpt))  — byte identity
//!   2. reconstructed W_i4 vs W_fp8 (f64): rel-L2, max-abs, per-row best-fit GAIN
//!   3. GEMV y = W·x: f64 reference from fp8 vs matvec_i4 — rel-L2 AND gain slope
//!   4. the SAME metrics for the vq3-derived chain (decode `.vq3` -> quant_i4), so
//!      the two `.i4` generations are compared head-to-head on one expert without
//!      regenerating the 365 GB set
//!   5. the outlier statistics (`amax/median`, `amax/p99.9`, spiky-row count) and the
//!      error split into BULK (|w| ≤ p99) and TAIL — the measurement that decides
//!      whether a quality drop is a defect or "fp8 rows are simply harder to quantize"
//!   6. `--scale-study`: sweep `s = α·amax/7` and score every α against the fp8 truth
//!      in BOTH weight and output space. `quant_i4` ships α = 1, which loads the
//!      quantizer at ~4.6σ where the MSE optimum for 15 levels is ~2.7σ; this finds
//!      the real optimum on real rows instead of assuming one.
//!
//! usage: i4_audit <artifact-dir> <fp8-dir> [--layer L] [--experts a,b,c]
//!                 [--scan] [--scale-study]
use anyhow::{Context, Result, anyhow, ensure};
use rivoli::format::{FormatMeta, I4Source, Safetensors, load_codebooks};
use rivoli::math::silu;
use rivoli::model::ModelConfig;
use rivoli::quant::{
    dequant_i4, i4_expert_stride, i4_row_bytes, i4_slot_offsets, matvec_i4, quant_i4, read_f32,
    vq_decode_proj, vq_expert, vq_expert_layout, vq_expert_stride,
};
use std::fs::File;
use std::os::unix::fs::FileExt;

const PROJ: [&str; 3] = ["gate_proj", "up_proj", "down_proj"];

/// `tests/kernel.rs::moe_i4_real_data_vs_fp8_ground_truth`'s `Lcg` seed. `make_x` is
/// bit-identical to that test's `Lcg::f`, so the CHAIN rows below are the SAME number
/// the test asserts on — which is the only way quoting them as its band is honest.
const CHAIN_SEED: u64 = 0x5A17;

/// The per-projection gain `quant_vq` imposes and `quant_i4` does not: `quant_vq`
/// refits its scale by least squares, which is MMSE-like and therefore shrinks by
/// `1 − relL2²`. Measured 0.9766 across every projection of every expert sampled
/// (predicted 0.9754 from its own rel-L2 — the agreement is why this is a property of
/// the estimator and not a coincidence of one artifact).
const VQ_GAIN: f32 = 0.9766;

/// Scale-sensitive agreement of `a` against reference `r`, all in f64.
/// Returns (rel_l2, max_abs, cosine, gain) where `gain` is the least-squares slope
/// `Σ a·r / Σ r·r` — 1.0 iff there is no systematic gain error. Cosine and gain
/// together separate "rotated" from "rescaled".
struct Agree {
    rel_l2: f64,
    max_abs: f64,
    cos: f64,
    gain: f64,
}
fn agree(a: &[f32], r: &[f32]) -> Agree {
    let (mut num, mut den, mut dot, mut na, mut nr, mut mx) = (0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
    for (&ai, &ri) in a.iter().zip(r) {
        let (ai, ri) = (ai as f64, ri as f64);
        let d = ai - ri;
        num += d * d;
        den += ri * ri;
        dot += ai * ri;
        na += ai * ai;
        nr += ri * ri;
        mx = mx.max(d.abs());
    }
    Agree {
        rel_l2: (num / den.max(f64::MIN_POSITIVE)).sqrt(),
        max_abs: mx,
        cos: dot / (na.sqrt() * nr.sqrt()).max(f64::MIN_POSITIVE),
        gain: dot / nr.max(f64::MIN_POSITIVE),
    }
}

/// f64 reference GEMV straight from the dense f32 weights (no quantized path).
fn matvec_f64(w: &[f32], x: &[f32], o_dim: usize, i_dim: usize) -> Vec<f32> {
    (0..o_dim)
        .map(|o| {
            let row = &w[o * i_dim..(o + 1) * i_dim];
            let mut acc = 0f64;
            for (i, &xi) in x.iter().enumerate() {
                acc += row[i] as f64 * xi as f64;
            }
            acc as f32
        })
        .collect()
}

/// What one projection's rows look like to an `amax/7` quantiser, and how much
/// precision the BULK of each row actually gets. Whole-row rel-L2 is dominated by the
/// large weights — it can improve while the many small weights, which is where decode
/// quality lives, get coarser. This separates the two.
struct Bulk {
    amax_over_med: f64,   // per-row amax / median|w|, mean over rows
    amax_over_p999: f64,  // per-row amax / p99.9|w|, mean over rows — the sharp outlier test
    spiky_rows: usize,    // rows where ≤4 weights reach half of amax (amax set by a handful)
    bulk_rel_l2: f64,     // rel-L2 of the reconstruction over |w| ≤ p99 ONLY
    tail_rel_l2: f64,     // rel-L2 over the |w| > p99 complement, for contrast
    levels_used: usize,   // distinct nibbles the bulk occupies, of 16
    step_mean: f64,       // mean per-row scale (the quantiser step)
    hist: [u64; 16],      // nibble histogram over the bulk
}

/// `w` is the reference (fp8 ground truth) the row statistics are taken from; `rec` is
/// a reconstruction to score against it; `scale` is the per-row step `rec` was built
/// with. Bulk/tail are split on the reference's own p99, so BOTH generations are scored
/// on the same positions.
fn bulk_stats(
    w: &[f32],
    rec: &[f32],
    packed: &[u8],
    scale: &[f32],
    o_dim: usize,
    i_dim: usize,
) -> Bulk {
    let rb = i4_row_bytes(i_dim);
    let (mut r_med, mut r_p999) = (0f64, 0f64);
    let (mut spiky, mut bn, mut bd, mut tn, mut td) = (0usize, 0f64, 0f64, 0f64, 0f64);
    let mut hist = [0u64; 16];
    let mut buf = vec![0f32; i_dim];
    for o in 0..o_dim {
        let (row, rrow) = (&w[o * i_dim..(o + 1) * i_dim], &rec[o * i_dim..(o + 1) * i_dim]);
        for (b, &v) in buf.iter_mut().zip(row) {
            *b = v.abs();
        }
        buf.sort_by(f32::total_cmp);
        let (med, amax) = (buf[i_dim / 2] as f64, buf[i_dim - 1] as f64);
        let p999 = buf[(i_dim * 999) / 1000] as f64;
        let p99 = buf[(i_dim * 99) / 100];
        r_med += if med > 0.0 { amax / med } else { 0.0 };
        r_p999 += if p999 > 0.0 { amax / p999 } else { 0.0 };
        // "amax set by a handful": how many weights reach half the row's extreme.
        let half = (amax * 0.5) as f32;
        if buf.iter().filter(|&&v| v >= half).count() <= 4 {
            spiky += 1;
        }
        let prow = &packed[o * rb..(o + 1) * rb];
        for i in 0..i_dim {
            let (a, b) = (row[i] as f64, rrow[i] as f64);
            let d = (b - a) * (b - a);
            if row[i].abs() <= p99 {
                bn += d;
                bd += a * a;
                let byte = prow[i >> 1];
                hist[(if i & 1 == 0 { byte & 0x0F } else { byte >> 4 }) as usize] += 1;
            } else {
                tn += d;
                td += a * a;
            }
        }
    }
    let n = o_dim as f64;
    Bulk {
        amax_over_med: r_med / n,
        amax_over_p999: r_p999 / n,
        spiky_rows: spiky,
        bulk_rel_l2: (bn / bd.max(f64::MIN_POSITIVE)).sqrt(),
        tail_rel_l2: (tn / td.max(f64::MIN_POSITIVE)).sqrt(),
        levels_used: hist.iter().filter(|&&c| c > 0).count(),
        step_mean: scale.iter().map(|&s| s as f64).sum::<f64>() / n,
        hist,
    }
}

/// Quantize one row to int4 at step `s` and score the reconstruction against `row`.
/// Returns `(sq_err, sq_ref, dot)` so a caller can pool them across rows into a
/// projection-wide rel-L2 and gain.
fn row_err(row: &[f32], s: f32) -> (f64, f64, f64) {
    let (mut e, mut r, mut d) = (0f64, 0f64, 0f64);
    for &w in row {
        let n = ((w / s).round() as i32).clamp(-8, 7) as f64 * s as f64;
        let w = w as f64;
        e += (n - w) * (n - w);
        r += w * w;
        d += n * w;
    }
    (e, r, d)
}

/// Does a better per-row STEP recover the ground the `.i4` set loses to `.vq3`?
///
/// `quant_i4` fixes `s = amax/7`: one outlier sets the step for the whole row. Two
/// candidate repairs, both measured here against the fp8 ground truth on real rows:
///
/// * **clip** — `s = α·amax/7` for α < 1, trading clipping of the extremes for a finer
///   step on the bulk. This is the percentile/absmax-clipping family.
/// * **LS refit** — round at `amax/7`, then set `s = Σ w·n / Σ n²`, the least-squares
///   optimum for the nibbles already chosen. `quant_vq` ALREADY does exactly this
///   (quant.rs: "refit the scale in closed form against the chosen entries"); `quant_i4`
///   does not. One line, no tunable.
///
/// Also reports each variant's GAIN, because that is the one systematic (as opposed to
/// noise) difference between the two `.i4` generations.
/// `yref` is the f64 GEMV of the unquantized rows against `x`, so each variant is also
/// scored in OUTPUT space — weight-space MSE is not the objective, and clipping trades
/// large-weight fidelity for bulk resolution in a way only the output error prices.
fn scale_study(w: &[f32], x: &[f32], yref: &[f32], o_dim: usize, i_dim: usize) {
    const ALPHAS: [f32; 13] = [
        1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4,
    ];
    const LS: usize = ALPHAS.len(); // LS refit
    const ORACLE: usize = ALPHAS.len() + 1; // per-row best α — the ceiling of a clip search
    let mut acc = [(0f64, 0f64, 0f64); ALPHAS.len() + 2]; // (sq_err, sq_ref, dot) in WEIGHT space
    let mut yacc = [(0f64, 0f64, 0f64); ALPHAS.len() + 2]; // the same in OUTPUT space
    let add = |t: &mut (f64, f64, f64), v: (f64, f64, f64)| {
        t.0 += v.0;
        t.1 += v.1;
        t.2 += v.2;
    };
    // Output-space error for one row at step `s`: y_o = Σ_i x_i·n_i·s against yref[o].
    let ydot = |row: &[f32], s: f32| -> f64 {
        row.iter()
            .zip(x)
            .map(|(&wi, &xi)| ((wi / s).round() as i32).clamp(-8, 7) as f64 * s as f64 * xi as f64)
            .sum()
    };
    for o in 0..o_dim {
        let row = &w[o * i_dim..(o + 1) * i_dim];
        let amax = row.iter().fold(0f32, |m, v| m.max(v.abs()));
        if amax <= 0.0 {
            continue;
        }
        let s0 = amax / 7.0;
        let yr = yref[o] as f64;
        let mut ytally = |k: usize, y: f64| {
            add(&mut yacc[k], ((y - yr) * (y - yr), yr * yr, y * yr));
        };
        let (mut best, mut best_i) = (f64::INFINITY, 0usize);
        for (i, &a) in ALPHAS.iter().enumerate() {
            let t = row_err(row, s0 * a);
            add(&mut acc[i], t);
            ytally(i, ydot(row, s0 * a));
            if t.0 < best {
                (best, best_i) = (t.0, i);
            }
        }
        // LS refit at the nibbles amax/7 already chose.
        let (mut num, mut den) = (0f64, 0f64);
        for &wi in row {
            let n = ((wi / s0).round() as i32).clamp(-8, 7) as f64;
            num += wi as f64 * n;
            den += n * n;
        }
        let sls = if den > 0.0 { (num / den) as f32 } else { s0 };
        add(&mut acc[LS], row_err(row, sls));
        ytally(LS, ydot(row, sls));
        add(&mut acc[ORACLE], row_err(row, s0 * ALPHAS[best_i]));
        ytally(ORACLE, ydot(row, s0 * ALPHAS[best_i]));
    }
    let show = |label: String, (e, r, d): (f64, f64, f64), (ye, yr, yd): (f64, f64, f64)| {
        println!(
            "        {label:<22} W relL2 {:.4}  W gain {:.4}   |   y relL2 {:.4}  y gain {:.4}",
            (e / r).sqrt(),
            d / r,
            (ye / yr).sqrt(),
            yd / yr
        );
    };
    for (i, a) in ALPHAS.iter().enumerate() {
        show(format!("s = {a:.2}·amax/7"), acc[i], yacc[i]);
    }
    show("LS refit (as quant_vq)".into(), acc[LS], yacc[LS]);
    show("per-row best α (oracle)".into(), acc[ORACLE], yacc[ORACLE]);
}

/// The audit's activation vector — bit-identical to `tests/kernel.rs::Lcg::f`
/// (uniform `[-1, 1)`, `>> 32`), so the numbers this tool prints are the SAME
/// statistic the GPU test's assertion band gates on. `silu` is not scale-invariant,
/// so a merely "similar" generator would quote a band for a different measurement.
///
/// Seeded by `wrapping_add`, NOT `seed | 1`: the latter maps 2n and 2n+1 to one
/// state, which silently gave gate and up the identical x.
fn make_x(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 32) as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Sweep every per-row f32 scale in the shipped `.i4` set looking for values the
/// decode path cannot survive: non-finite (`0 · inf = NaN` in `matvec_i4`/`dot_i4_wave`,
/// which apply the scale OUTSIDE the dot), zero, negative, or the `amax == 0` sentinel
/// `1.0` (a dead row). Cheap: scales are 40 KB per expert, so this preads ~10 MB/layer.
fn scan_scales(art: &str, cfg: &ModelConfig) -> Result<()> {
    let (h, m, ne) = (cfg.hidden, cfg.moe_inter, cfg.n_experts);
    let off = i4_slot_offsets(h, m);
    let stride = i4_expert_stride(h, m);
    let dims = vq_expert_layout(h, m);
    let (mut nonfinite, mut zero, mut neg, mut ones, mut total) = (0u64, 0u64, 0u64, 0u64, 0u64);
    let (mut lo, mut hi) = (f32::INFINITY, 0f32);
    let mut worst = (0usize, 0usize, 0usize, 0usize, 0f32); // layer, expert, proj, row, scale
    for l in cfg.dense_layers..cfg.n_layers {
        let f = File::open(format!("{art}/L{l:02}.i4"))?;
        for e in 0..=ne {
            for (k, &(o_dim, _)) in dims.iter().enumerate() {
                let mut b = vec![0u8; o_dim * 4];
                f.read_exact_at(&mut b, (e * stride + off[k * 2 + 1]) as u64)?;
                for (o, s) in read_f32(&b).into_iter().enumerate() {
                    total += 1;
                    if !s.is_finite() {
                        nonfinite += 1;
                        worst = (l, e, k, o, s);
                    } else if s == 0.0 {
                        zero += 1;
                    } else if s < 0.0 {
                        neg += 1;
                    } else {
                        if s == 1.0 {
                            ones += 1;
                        }
                        if s < lo {
                            lo = s;
                        }
                        if s > hi {
                            hi = s;
                            worst = (l, e, k, o, s);
                        }
                    }
                }
            }
        }
        eprint!("\rscanned L{l:02}  ");
    }
    eprintln!();
    println!(
        "scales: {total} total, nonfinite {nonfinite}, zero {zero}, negative {neg}, ==1.0 (amax==0 dead row) {ones}"
    );
    println!(
        "        min {lo:.6e}  max {hi:.6e}  (max at layer {} expert {} proj {} row {}, s={:.6e})",
        worst.0, worst.1, worst.2, worst.3, worst.4
    );
    println!("        max|W| implied by the largest scale = 7·s = {:.4e}", hi * 7.0);
    Ok(())
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let art = args.next().context("usage: i4_audit <artifact-dir> <fp8-dir> [--layer L] [--experts a,b,c] [--scan]")?;
    let fp8 = args.next().context("missing <fp8-dir>")?;
    let (mut layer, mut experts) = (3usize, vec![0usize, 7, 128, 255]);
    let (mut scan, mut study) = (false, false);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--scan" => scan = true,
            "--scale-study" => study = true,
            "--layer" => layer = args.next().context("--layer L")?.parse()?,
            "--experts" => {
                experts = args
                    .next()
                    .context("--experts a,b,c")?
                    .split(',')
                    .map(|s| s.parse::<usize>().map_err(|e| anyhow!("{e}")))
                    .collect::<Result<_>>()?
            }
            _ => return Err(anyhow!("unknown arg {a}")),
        }
    }

    let cfg = ModelConfig::load(&art)?;
    if scan {
        return scan_scales(&art, &cfg);
    }
    let (h, m, ne) = (cfg.hidden, cfg.moe_inter, cfg.n_experts);
    // `e == ne` is the shared expert; anything ABOVE it would read routed block `e`'s
    // bytes and score them against the SHARED expert's fp8 weights — a full row of
    // garbage metrics plus a `DISK != quant_i4(fp8)` marker blaming the artifact for
    // the operator's typo. Fail hard instead (as `fp8_to_i4` does for its layer range).
    ensure!(
        experts.iter().all(|&e| e <= ne),
        "--experts: {ne} is the shared expert and the largest valid index"
    );
    let block = FormatMeta::load(&art)?.fp8_block;
    // The checkpoint is taken on faith otherwise: a DIFFERENT revision of the same
    // model passes `dequant_fp8`'s shape checks, turns every byte-identity check false,
    // and makes the report read "the shipped .i4 is defective" when the argument is.
    match I4Source::load(&art)? {
        Some(p) if std::fs::canonicalize(&fp8).map(|c| c.display().to_string()) .map(|c| c != p.src).unwrap_or(true) => {
            eprintln!("WARNING: artifact was built from {} — auditing against {fp8}", p.src)
        }
        None => eprintln!("WARNING: artifact carries no i4_source stamp; provenance unverified"),
        _ => {}
    }
    let off = i4_slot_offsets(h, m);
    let i4_stride = i4_expert_stride(h, m);
    let vq_stride = vq_expert_stride(h, m);
    let dims = vq_expert_layout(h, m);
    let cbs = load_codebooks(&art)?;

    let src = Safetensors::open_dir(&fp8).context("open fp8 checkpoint")?;
    let i4f = File::open(format!("{art}/L{layer:02}.i4"))?;
    let vqf = File::open(format!("{art}/L{layer:02}.vq3"))?;

    println!("layer {layer}  hidden {h}  moe_inter {m}  fp8_block {block}");
    println!(
        "{:<6} {:<10} | {:>10} {:>10} {:>9} {:>9} | {:>10} {:>9} {:>9} |",
        "expert", "proj", "W relL2", "W maxerr", "W cos", "W gain", "y relL2", "y cos", "y gain"
    );
    // The exact verdict, and the one worth an exit status: everything below is a
    // statistic with a tolerance, but "the disk is byte-for-byte what the converter
    // would write from this checkpoint" either holds or it does not.
    let mut mismatches = 0usize;

    for &e in &experts {
        // The `.i4` block on disk for this expert (headerless, blocks from 0).
        let mut i4b = vec![0u8; i4_stride];
        i4f.read_exact_at(&mut i4b, (e * i4_stride) as u64)
            .with_context(|| format!("read .i4 expert {e}"))?;
        // The `.vq3` block for the same expert (VQ_ALIGN header).
        let mut vqb = vec![0u8; vq_stride];
        vqf.read_exact_at(&mut vqb, (rivoli::quant::VQ_ALIGN + e * vq_stride) as u64)
            .with_context(|| format!("read .vq3 expert {e}"))?;
        let vqp = vq_expert(&vqb, 0, h, m);

        let base = if e < ne {
            format!("model.layers.{layer}.mlp.experts.{e}")
        } else {
            format!("model.layers.{layer}.mlp.shared_experts")
        };

        // Kept per projection so the whole-expert chain below can be run three ways.
        let (mut wrefs, mut i4s, mut olds, mut vqs) = (vec![], vec![], vec![], vec![]);
        for (k, (&(o_dim, i_dim), pname)) in dims.iter().zip(PROJ).enumerate() {
            // ── independent ground truth: the fp8 checkpoint ─────────────────
            let wref = src.dequant_fp8(&format!("{base}.{pname}"), o_dim, i_dim, block)?;

            // ── what the artifact actually holds ─────────────────────────────
            let rb = i4_row_bytes(i_dim);
            let packed = &i4b[off[k * 2]..off[k * 2] + o_dim * rb];
            let scale = read_f32(&i4b[off[k * 2 + 1]..off[k * 2 + 1] + o_dim * 4]);

            // (1) byte identity: does the disk match a fresh quant_i4 of the fp8?
            let (p2, s2) = quant_i4(&wref, o_dim, i_dim);
            let bytes_match = p2 == packed;
            let scales_match = s2 == scale;
            mismatches += usize::from(!(bytes_match && scales_match));

            // (2) reconstructed matrix vs fp8, scale-sensitive
            let wi4 = dequant_i4(packed, &scale, o_dim, i_dim);
            let aw = agree(&wi4, &wref);

            // (3) GEMV, f64 reference from fp8 vs the shipped int4 path
            let x = make_x(i_dim, 0xB0BA_u64.wrapping_add((e * 7 + k) as u64));
            let yref = matvec_f64(&wref, &x, o_dim, i_dim);
            let mut yi4 = vec![0f32; o_dim];
            matvec_i4(&mut yi4, &x, packed, &scale, o_dim, i_dim);
            let ay = agree(&yi4, &yref);

            println!(
                "{:<6} {:<10} | {:>10.4e} {:>10.3e} {:>9.6} {:>9.6} | {:>10.4e} {:>9.6} {:>9.6} |{}",
                e, pname, aw.rel_l2, aw.max_abs, aw.cos, aw.gain, ay.rel_l2, ay.cos, ay.gain,
                if bytes_match && scales_match { "" } else { "  <<< DISK != quant_i4(fp8)" }
            );

            // ── the OLD chain, same expert: fp8 -> vq3 -> int4 ───────────────
            let wvq = vq_decode_proj(&vqp[k], &cbs[k]);
            let (op, os) = quant_i4(&wvq, o_dim, i_dim);
            let wold = dequant_i4(&op, &os, o_dim, i_dim);
            let aw_old = agree(&wold, &wref);
            let mut yold = vec![0f32; o_dim];
            matvec_i4(&mut yold, &x, &op, &os, o_dim, i_dim);
            let ay_old = agree(&yold, &yref);
            // and the vq3 path itself (the control mode's arithmetic)
            let avq = agree(&wvq, &wref);
            let yvq = matvec_f64(&wvq, &x, o_dim, i_dim);
            let ayvq = agree(&yvq, &yref);
            println!(
                "{:<6} {:<10} | {:>10.4e} {:>10.3e} {:>9.6} {:>9.6} | {:>10.4e} {:>9.6} {:>9.6} |",
                "", "  ^old i4", aw_old.rel_l2, aw_old.max_abs, aw_old.cos, aw_old.gain,
                ay_old.rel_l2, ay_old.cos, ay_old.gain
            );
            println!(
                "{:<6} {:<10} | {:>10.4e} {:>10.3e} {:>9.6} {:>9.6} | {:>10.4e} {:>9.6} {:>9.6} |",
                "", "  ^vq3", avq.rel_l2, avq.max_abs, avq.cos, avq.gain,
                ayvq.rel_l2, ayvq.cos, ayvq.gain
            );
            // ── bulk vs tail: does the fp8 chain's LARGER amax coarsen the step and
            //    cost precision on the many small weights, where decode quality lives?
            //    Both generations are scored on the SAME positions (the fp8 row's own
            //    p99), so the split cannot flatter either one.
            let bn = bulk_stats(&wref, &wi4, packed, &scale, o_dim, i_dim);
            let bo = bulk_stats(&wref, &wold, &op, &os, o_dim, i_dim);
            for (label, b) in [("new i4", &bn), ("old i4", &bo)] {
                println!(
                    "{:<6} BULK {:<8}| amax/med {:>5.1}  amax/p99.9 {:>5.2}  spiky rows {:>5}/{o_dim}  \
                     step {:>9.3e}  bulk relL2 {:>7.4}  tail relL2 {:>7.4}  levels {:>2}/16",
                    e, label, b.amax_over_med, b.amax_over_p999, b.spiky_rows, b.step_mean,
                    b.bulk_rel_l2, b.tail_rel_l2, b.levels_used
                );
            }
            println!("{:<6} HIST new  | {:?}", e, bn.hist);
            println!("{:<6} HIST old  | {:?}", e, bo.hist);
            println!(
                "{:<6} STEP ratio| new/old = {:.4}  (>1 ⇒ the fp8 amax coarsened the step)",
                e,
                bn.step_mean / bo.step_mean
            );
            if study {
                println!("      SCALE STUDY (fp8 rows, expert {e} {pname}) — vq3 reaches relL2 {:.4}:", avq.rel_l2);
                scale_study(&wref, &x, &yref, o_dim, i_dim);
            }

            wrefs.push(wref);
            i4s.push((packed.to_vec(), scale));
            olds.push((op, os));
            vqs.push(wvq);
        }

        // ── the WHOLE expert: down(silu(gate·x) ⊙ up·x), the quantity the kernel
        //    actually produces, against the f64 fp8 chain. This is what the GPU test
        //    asserts on, so measure the bounds here rather than guessing them.
        // The GPU test's seed, not a fresh one per expert: `tests/kernel.rs` quotes
        // this tool's CHAIN row as its assertion band, and rel-L2 through silu varies
        // by ~15% across x draws — enough that a merely same-DISTRIBUTION x quotes a
        // band the test does not actually sit in. Same generator AND same seed.
        let x = make_x(h, CHAIN_SEED);
        let chain_ref = {
            let g = matvec_f64(&wrefs[0], &x, m, h);
            let u = matvec_f64(&wrefs[1], &x, m, h);
            let hv: Vec<f32> = (0..m).map(|j| silu(g[j]) * u[j]).collect();
            matvec_f64(&wrefs[2], &hv, h, m)
        };
        let chain_i4 = |q: &[(Vec<u8>, Vec<f32>); 3]| -> Vec<f32> {
            let (mut g, mut u) = (vec![0f32; m], vec![0f32; m]);
            matvec_i4(&mut g, &x, &q[0].0, &q[0].1, m, h);
            matvec_i4(&mut u, &x, &q[1].0, &q[1].1, m, h);
            let hv: Vec<f32> = (0..m).map(|j| silu(g[j]) * u[j]).collect();
            let mut o = vec![0f32; h];
            matvec_i4(&mut o, &hv, &q[2].0, &q[2].1, h, m);
            o
        };
        let newq: [(Vec<u8>, Vec<f32>); 3] = core::array::from_fn(|k| i4s[k].clone());
        let oldq: [(Vec<u8>, Vec<f32>); 3] = core::array::from_fn(|k| olds[k].clone());
        let cn = agree(&chain_i4(&newq), &chain_ref);
        let co = agree(&chain_i4(&oldq), &chain_ref);
        // PRE-FLIGHT for the attenuation arm. The shipped nibbles with every stored
        // per-row scale multiplied by VQ_GAIN — vq3's shrink WITHOUT vq3's error. If
        // this does not land the chain gain inside the old set's band, the arm does not
        // reproduce what it claims to and the device block would measure nothing.
        //
        // Multiplies the STORED scale only; the nibbles are the ones `amax/7` already
        // chose. Rounding at `VQ_GAIN·amax/7` instead would be the α = 0.977 CLIP arm,
        // a different experiment (finer step, gain still ~1).
        let gainq: [(Vec<u8>, Vec<f32>); 3] = core::array::from_fn(|k| {
            (i4s[k].0.clone(), i4s[k].1.iter().map(|s| s * VQ_GAIN).collect())
        });
        let cg = agree(&chain_i4(&gainq), &chain_ref);
        let cv = {
            let g = matvec_f64(&vqs[0], &x, m, h);
            let u = matvec_f64(&vqs[1], &x, m, h);
            let hv: Vec<f32> = (0..m).map(|j| silu(g[j]) * u[j]).collect();
            agree(&matvec_f64(&vqs[2], &hv, h, m), &chain_ref)
        };
        // `max_err / max|ref|` is the per-ELEMENT statistic the aggregate rel-L2 cannot
        // see: corruption confined to a few percent of output rows moves rel-L2 by less
        // than its own tolerance, but moves this. The GPU test asserts on it.
        let ymax = chain_ref.iter().fold(0f64, |m, &v| m.max(v.abs() as f64));
        for (label, a) in [("new i4", cn), ("old i4", co), ("vq3", cv), ("new×gain", cg)] {
            println!(
                "{:<6} CHAIN {:<7}| relL2 {:>7.4}  gain {:>7.4}  cos {:>8.6}  maxerr/max|ref| {:>7.4}",
                e, label, a.rel_l2, a.gain, a.cos, a.max_abs / ymax
            );
        }
    }
    println!(
        "\nBYTE IDENTITY: {} of {} projections differ from quant_i4(dequant_fp8(ckpt))",
        mismatches,
        experts.len() * 3
    );
    ensure!(mismatches == 0, "the shipped .i4 is NOT what this checkpoint quantizes to");
    Ok(())
}