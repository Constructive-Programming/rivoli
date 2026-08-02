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
//!   4. the same metrics for `.vq3` decoded at the same coordinates — `--mode int3-vq`'s
//!      arithmetic, the other shipped routed-expert format, scored against one truth
//!   5. the outlier statistics (`amax/median`, `amax/p99.9`, spiky-row count) and the
//!      error split into BULK (|w| ≤ p99) and TAIL — the measurement that decides
//!      whether a quality drop is a defect or "fp8 rows are simply harder to quantize"
//!   6. `--scale-study`: score every candidate quantiser step against the fp8 truth in
//!      BOTH weight and output space — a sweep of PER-ROW steps `s = α·amax/7`, the LS
//!      refit, the per-row oracle, and the shipped GROUP-`I4_GROUP` quantiser. The last
//!      row against the `α = 1.00` row is the head-to-head that justified moving off
//!      per-row scales. This mode reads ONLY the fp8 checkpoint (the study is a
//!      property of the weights and the quantiser), so it runs before a rebuild and
//!      against an artifact of any generation.
//!
//! `--help` is the flag reference. What this tool NO LONGER carries: the
//! `fp8 → vq3 → int4` "old i4" rows and the `VQ_GAIN` attenuation pre-flight built on
//! them, both removed 2026-08-01. They head-to-headed the shipped set against a
//! generation nothing can produce any more (`bin/vq3_to_i4` is deleted —
//! docs/reference/architecture.md §11), and the branch-gain hypothesis they fed is
//! falsified outright (docs/investigations/int4-scales.md §3, "Falsified: branch gain":
//! attenuation is monotonically harmful, no interior optimum). Every number they printed
//! is tabulated in docs/measurement/benchmarks.md.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use rivoli::artifact::format::{FormatMeta, I4Source, Safetensors, load_codebooks};
use rivoli::artifact::model::ModelConfig;
use rivoli::artifact::quant::{
    I4_GROUP, PROJ, dequant_i4, expert_base, i4_expert_stride, i4_groups, i4_row_bytes,
    i4_slot_offsets, matvec_i4, quant_i4, read_f32, vq_decode_proj, vq_expert, vq_expert_layout,
    vq_expert_stride,
};
use rivoli::math::silu;
use std::fs::File;
use std::os::unix::fs::FileExt;

// NOTE: doc comments on the FIELDS below are USER-FACING — clap renders them as `--help`.
// Rationale for the code goes in `//` comments like this one, which clap ignores. The
// hand-rolled loop this replaced kept a `usage:` string in a `.context()` on the first
// positional, and it had already drifted: it named neither `--verify` nor `--xcheck`, the
// two wide modes docs/investigations/int4-scales.md §8 keeps this tool for.
#[derive(Parser)]
#[command(
    name = "i4_audit",
    about = "Scale-sensitive audit of the .i4 expert path against the original fp8 checkpoint"
)]
struct Args {
    /// The artifact directory to audit: manifest.json, `L{ll}.i4`, `L{ll}.vq3`,
    /// codebooks.f32. Its `i4_source` stamp is checked against <FP8_DIR> and a
    /// disagreement is reported as a WARNING — auditing the wrong checkpoint turns every
    /// byte-identity check false and reads as "the shipped .i4 is defective".
    artifact_dir: String,

    /// The fp8 checkpoint the artifact was built from — the INDEPENDENT ground truth, in
    /// f64. Every mode but `--xcheck` reads it.
    fp8_dir: String,

    /// Sweep every f32 group scale in the WHOLE `.i4` set for values the decode path
    /// cannot survive: non-finite (`0·inf = NaN` in the dot), zero, negative, or the
    /// `amax == 0` sentinel 1.0. Reads scales only (~2 MB/expert), so it covers all
    /// layers cheaply, and reports the observed range.
    #[arg(long)]
    scan: bool,

    /// Wide byte-identity sweep — `dequant_fp8` + `quant_i4` + compare, and nothing else,
    /// so it runs ~10× faster per projection than the default report and covers hundreds
    /// (every 6th layer × every 31st expert × 3) instead of tens. Exits non-zero on any
    /// mismatch. An error confined to particular layers or one converter thread's chunk
    /// is exactly what a narrow sample misses.
    #[arg(long)]
    verify: bool,

    /// Cross-check `.i4` against `.vq3` at the SAME artifact coordinates — the one check
    /// here that can fail on a layer/expert MAPPING error, because it touches no fp8, no
    /// `dequant_fp8` and no `model.layers.{l}` string. Two quantizations of one matrix
    /// score cos ≈ 0.97; two unrelated ones score cos ≈ 0. Exits non-zero below 0.90.
    #[arg(long)]
    xcheck: bool,

    /// Score every candidate quantiser step against the fp8 truth in BOTH weight and
    /// output space: the per-row `s = α·amax/7` sweep, the LS refit `quant_vq` uses, the
    /// per-row oracle, and the shipped GROUP-I4_GROUP quantiser. Reads ONLY the fp8
    /// checkpoint, so it answers "is a rebuild worth buying?" BEFORE the rebuild and runs
    /// against an artifact of any generation.
    #[arg(long)]
    scale_study: bool,

    /// The layer to audit, for the default per-projection report and `--scale-study`.
    /// `--scan`, `--verify` and `--xcheck` walk their own layer sets and ignore it.
    #[arg(long, value_name = "L", default_value_t = 3)]
    layer: usize,

    /// Experts to audit, comma-separated, for the same two modes. `n_experts` is the
    /// SHARED expert and the largest valid index — anything above it would score routed
    /// block `e`'s bytes against the shared expert's weights, so it is rejected.
    // `default_value` (one string) rather than `default_values_t` (four numbers) only so
    // that `--help` shows the default in the same comma syntax the flag is typed in;
    // clap runs the delimiter over the default too, so both spell the same four values.
    #[arg(
        long,
        value_name = "A,B,C",
        value_delimiter = ',',
        default_value = "0,7,128,255"
    )]
    experts: Vec<usize>,
}

/// `tests/kernel.rs::moe_i4_real_data_vs_fp8_ground_truth`'s `Lcg` seed. `make_x` is
/// bit-identical to that test's `Lcg::f`, so the CHAIN rows below are the SAME number
/// the test asserts on — which is the only way quoting them as its band is honest.
const CHAIN_SEED: u64 = 0x5A17;

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
    amax_over_med: f64,  // per-row amax / median|w|, mean over rows
    amax_over_p999: f64, // per-row amax / p99.9|w|, mean over rows — the sharp outlier test
    spiky_rows: usize,   // rows where ≤4 weights reach half of amax (amax set by a handful)
    bulk_rel_l2: f64,    // rel-L2 of the reconstruction over |w| ≤ p99 ONLY
    tail_rel_l2: f64,    // rel-L2 over the |w| > p99 complement, for contrast
    levels_used: usize,  // distinct nibbles the bulk occupies, of 16
    step_mean: f64,      // mean quantiser step (over every stored scale)
    dead_rows: usize,    // rows where >50% of weights rounded to nibble 8 (zero)
    hist: [u64; 16],     // nibble histogram over the bulk
}

/// `w` is the reference (fp8 ground truth) the row statistics are taken from; `rec` is
/// a reconstruction to score against it; `scale` is the step array `rec` was built with
/// (`o_dim · i4_groups(i_dim)` under group scales). Bulk/tail are split on the
/// REFERENCE's own p99, not on the reconstruction's, so the split cannot flatter the
/// thing being scored — and two reconstructions of one matrix land on the same positions.
///
/// `dead_rows` is the headline number for the per-row → group change: under one scale
/// per 6144-wide row a single outlier rounded the bulk to zero, and 603 of 6144 rows of
/// L03 e0 down_proj ended past 50% zeros. A group scale should collapse that count.
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
    let mut dead = 0usize;
    let mut hist = [0u64; 16];
    let mut buf = vec![0f32; i_dim];
    for o in 0..o_dim {
        let (row, rrow) = (
            &w[o * i_dim..(o + 1) * i_dim],
            &rec[o * i_dim..(o + 1) * i_dim],
        );
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
        let mut zeros = 0usize;
        for i in 0..i_dim {
            let (a, b) = (row[i] as f64, rrow[i] as f64);
            let d = (b - a) * (b - a);
            let byte = prow[i >> 1];
            let nib = (if i & 1 == 0 { byte & 0x0F } else { byte >> 4 }) as usize;
            zeros += usize::from(nib == 8);
            if row[i].abs() <= p99 {
                bn += d;
                bd += a * a;
                hist[nib] += 1;
            } else {
                tn += d;
                td += a * a;
            }
        }
        if zeros * 2 > i_dim {
            dead += 1;
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
        step_mean: scale.iter().map(|&s| s as f64).sum::<f64>() / scale.len() as f64,
        dead_rows: dead,
        hist,
    }
}

/// Quantize one row to int4 at step `s` and score the reconstruction against `row`.
/// Returns `(sq_err, sq_ref, dot, zeros)` so a caller can pool them across rows into a
/// projection-wide rel-L2, gain, and zero fraction.
///
/// `zeros` is the count that rounded to the middle level — weights the quantiser threw
/// away entirely. It is the direct measure of the per-row pathology: one outlier sets
/// `s = amax/7` for the whole row and the bulk lands on zero.
fn row_err(row: &[f32], s: f32) -> (f64, f64, f64, u64) {
    let (mut e, mut r, mut d, mut z) = (0f64, 0f64, 0f64, 0u64);
    for &w in row {
        let q = ((w / s).round() as i32).clamp(-8, 7);
        z += u64::from(q == 0);
        let n = q as f64 * s as f64;
        let w = w as f64;
        e += (n - w) * (n - w);
        r += w * w;
        d += n * w;
    }
    (e, r, d, z)
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
    const GROUP: usize = ALPHAS.len() + 2; // the SHIPPED quantiser: amax/7 per I4_GROUP
    const NVAR: usize = ALPHAS.len() + 3;
    let mut acc = [(0f64, 0f64, 0f64, 0u64); NVAR]; // (sq_err, sq_ref, dot, zeros), WEIGHT space
    let mut yacc = [(0f64, 0f64, 0f64); NVAR]; // (sq_err, sq_ref, dot) in OUTPUT space
    let add = |t: &mut (f64, f64, f64, u64), v: (f64, f64, f64, u64)| {
        t.0 += v.0;
        t.1 += v.1;
        t.2 += v.2;
        t.3 += v.3;
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
            let t = &mut yacc[k];
            (t.0, t.1, t.2) = (t.0 + (y - yr) * (y - yr), t.1 + yr * yr, t.2 + y * yr);
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
        // The SHIPPED quantiser: `amax/7` recomputed per I4_GROUP columns. Every α row
        // above is a PER-ROW step, so this is the head-to-head the format change rests
        // on — same rows, same x, same metrics.
        let (mut gw, mut gy) = ((0f64, 0f64, 0f64, 0u64), 0f64);
        for (g, seg) in row.chunks(I4_GROUP).enumerate() {
            let gmax = seg.iter().fold(0f32, |m, v| m.max(v.abs()));
            let sg = if gmax > 0.0 { gmax / 7.0 } else { s0 };
            add(&mut gw, row_err(seg, sg));
            gy += seg
                .iter()
                .zip(&x[g * I4_GROUP..])
                .map(|(&wi, &xi)| {
                    ((wi / sg).round() as i32).clamp(-8, 7) as f64 * sg as f64 * xi as f64
                })
                .sum::<f64>();
        }
        add(&mut acc[GROUP], gw);
        ytally(GROUP, gy);
    }
    let total = (o_dim * i_dim) as f64;
    let show = |label: String,
                (e, r, d, z): (f64, f64, f64, u64),
                (ye, yr, yd): (f64, f64, f64)| {
        println!(
            "        {label:<22} W relL2 {:.4}  W gain {:.4}  zeros {:>6.2}%   |   y relL2 {:.4}  y gain {:.4}",
            (e / r).sqrt(),
            d / r,
            100.0 * z as f64 / total,
            (ye / yr).sqrt(),
            yd / yr
        );
    };
    for (i, a) in ALPHAS.iter().enumerate() {
        show(format!("s = {a:.2}·amax/7"), acc[i], yacc[i]);
    }
    show("LS refit (as quant_vq)".into(), acc[LS], yacc[LS]);
    show("per-row best α (oracle)".into(), acc[ORACLE], yacc[ORACLE]);
    show(format!("GROUP-{I4_GROUP} amax/7"), acc[GROUP], yacc[GROUP]);
}

/// `--scale-study` on its own: score every candidate step against the fp8 checkpoint
/// WITHOUT reading the artifact. The study is a property of the weights and the
/// quantiser, never of the shipped bytes — and keeping it independent is what lets it
/// answer "is this format worth rebuilding 386 GB for?" BEFORE the rebuild, and lets it
/// run at all while the `.i4` set on disk is a different generation's format.
fn scale_study_only(
    fp8: &str,
    cfg: &ModelConfig,
    block: usize,
    layer: usize,
    experts: &[usize],
) -> Result<()> {
    let (h, m, ne) = (cfg.hidden, cfg.moe_inter, cfg.n_experts);
    let src = Safetensors::open_dir(fp8).context("open fp8 checkpoint")?;
    println!(
        "layer {layer}  hidden {h}  moe_inter {m}  fp8_block {block}  (fp8 only — artifact not read)"
    );
    for &e in experts {
        ensure!(
            e <= ne,
            "--experts: {ne} is the shared expert and the largest valid index"
        );
        let base = expert_base(layer, e, ne);
        for (k, (&(o_dim, i_dim), pname)) in vq_expert_layout(h, m).iter().zip(PROJ).enumerate() {
            let wref = src.dequant_fp8(&format!("{base}.{pname}"), o_dim, i_dim, block)?;
            let x = make_x(i_dim, 0xB0BA_u64.wrapping_add((e * 7 + k) as u64));
            let yref = matvec_f64(&wref, &x, o_dim, i_dim);
            println!("      SCALE STUDY (fp8 rows, expert {e} {pname}, {o_dim}×{i_dim}):");
            scale_study(&wref, &x, &yref, o_dim, i_dim);
        }
    }
    Ok(())
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
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 32) as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Byte identity over a WIDE sample: `dequant_fp8` + `quant_i4` + compare, and nothing
/// else, so it runs ~10x faster than the full audit and can cover hundreds of
/// projections instead of tens. An error confined to a subset — particular layers, a
/// shape, a thread's chunk of the converter's work split — is entirely consistent with
/// a narrow sample passing, which is the hole this closes.
fn verify_wide(
    art: &str,
    fp8: &str,
    cfg: &ModelConfig,
    layers: &[usize],
    experts: &[usize],
) -> Result<()> {
    let (h, m, ne) = (cfg.hidden, cfg.moe_inter, cfg.n_experts);
    let block = FormatMeta::load(art)?.fp8_block;
    let (stride, off) = (i4_expert_stride(h, m), i4_slot_offsets(h, m));
    let src = Safetensors::open_dir(fp8).context("open fp8 checkpoint")?;
    let (mut ok, mut bad) = (0usize, 0usize);
    for &l in layers {
        let f = File::open(format!("{art}/L{l:02}.i4"))?;
        for &e in experts {
            let mut blk = vec![0u8; stride];
            f.read_exact_at(&mut blk, (e * stride) as u64)?;
            let base = expert_base(l, e, ne);
            for (k, (&(o_dim, i_dim), proj)) in vq_expert_layout(h, m).iter().zip(PROJ).enumerate()
            {
                let w = src.dequant_fp8(&format!("{base}.{proj}"), o_dim, i_dim, block)?;
                let (wp, ws) = quant_i4(&w, o_dim, i_dim);
                let po = off[k * 2];
                let gp = &blk[po..po + o_dim * i4_row_bytes(i_dim)];
                let so = off[k * 2 + 1];
                let gs = read_f32(&blk[so..so + o_dim * 4]);
                if wp == gp && ws == gs {
                    ok += 1;
                } else {
                    bad += 1;
                    let rows = wp
                        .chunks_exact(i4_row_bytes(i_dim))
                        .zip(gp.chunks_exact(i4_row_bytes(i_dim)))
                        .filter(|(a, b)| a != b)
                        .count();
                    println!(
                        "MISMATCH L{l:02} e{e} {proj}: {rows}/{o_dim} rows, scales_eq={}",
                        ws == gs
                    );
                }
            }
        }
        eprint!("\rverified L{l:02}  ");
    }
    eprintln!();
    println!(
        "WIDE VERIFY: {ok} projections bit-exact, {bad} mismatched ({} layers x {} experts x 3)",
        layers.len(),
        experts.len()
    );
    ensure!(
        bad == 0,
        "{bad} projections differ from quant_i4(dequant_fp8(ckpt))"
    );
    Ok(())
}

/// Cross-check `.i4` against `.vq3` at the SAME artifact coordinates — the one
/// comparison in this tool that can actually fail on a mapping error.
///
/// Every other check here re-dequantises fp8, which runs the same tensor-name and
/// layer resolution `fp8_to_i4` used: an artifact layer built from the WRONG checkpoint
/// layer would be bit-exactly reproduced by the audit and score perfectly. This path
/// touches no fp8, no `dequant_fp8`, and no `model.layers.{l}` string. It reads
/// `L{l}.i4` and `L{l}.vq3` at the same (layer, expert, projection) and asks whether
/// they describe the same matrix. `.vq3` came from a separate tool with its own
/// iteration, and the int3-vq model built on it decodes coherently — so if the two
/// agree, the `.i4` at that coordinate holds that coordinate's weights.
///
/// Expected: both are quantizations of one matrix, so `rel-L2 ≈ sqrt(0.205² + 0.159²)
/// ≈ 0.26` and `cos ≈ 0.97`. A mapping error gives two UNRELATED matrices: `cos ≈ 0`,
/// `rel-L2 ≈ 1.41`. The gap between those outcomes is enormous, so this needs no
/// delicate tolerance — and a cluster of low cells past some index is an off-by-one at
/// a boundary rather than a uniform mis-mapping.
fn xcheck(art: &str, cfg: &ModelConfig, layers: &[usize], experts: &[usize]) -> Result<()> {
    let (h, m) = (cfg.hidden, cfg.moe_inter);
    let (i4_stride, off) = (i4_expert_stride(h, m), i4_slot_offsets(h, m));
    let vq_stride = vq_expert_stride(h, m);
    let cbs = load_codebooks(art)?;
    let mut worst = (1.0f64, 0usize, 0usize, "");
    let mut low = 0usize;
    println!(
        "{:<6}{:<6}{:>12}{:>10}{:>10}",
        "layer", "expert", "proj", "cos", "relL2"
    );
    for &l in layers {
        let i4f = File::open(format!("{art}/L{l:02}.i4"))?;
        let vqf = File::open(format!("{art}/L{l:02}.vq3"))?;
        for &e in experts {
            let mut i4b = vec![0u8; i4_stride];
            i4f.read_exact_at(&mut i4b, (e * i4_stride) as u64)?;
            let vqb = read_vq_block(&vqf, e, vq_stride)?;
            let vqp = vq_expert(&vqb, 0, h, m);
            for (k, (&(o_dim, i_dim), proj)) in vq_expert_layout(h, m).iter().zip(PROJ).enumerate()
            {
                let packed = &i4b[off[k * 2]..off[k * 2] + o_dim * i4_row_bytes(i_dim)];
                let scale = read_f32(&i4b[off[k * 2 + 1]..off[k * 2 + 1] + o_dim * 4]);
                let wi4 = dequant_i4(packed, &scale, o_dim, i_dim);
                let wvq = vq_decode_proj(&vqp[k], &cbs[k]);
                let a = agree(&wi4, &wvq);
                if a.cos < 0.90 {
                    low += 1;
                    println!(
                        "{l:<6}{e:<6}{proj:>12}{:>10.4}{:>10.4}   <<< LOW",
                        a.cos, a.rel_l2
                    );
                } else if a.cos < worst.0 {
                    worst = (a.cos, l, e, proj);
                }
            }
        }
        eprint!("\rxcheck L{l:02}  ");
    }
    eprintln!();
    let n = layers.len() * experts.len() * 3;
    println!(
        "XCHECK: {n} projections ({} layers x {} experts x 3), {low} with cos < 0.90; \
         worst good cell cos {:.4} at L{:02} e{} {}",
        layers.len(),
        experts.len(),
        worst.0,
        worst.1,
        worst.2,
        worst.3
    );
    ensure!(
        low == 0,
        "{low} projections do not describe the same matrix as .vq3 — mapping error"
    );
    Ok(())
}

/// Sweep every f32 group scale in the shipped `.i4` set looking for values the decode
/// path cannot survive: non-finite (`0 · inf = NaN` in `matvec_i4`/`dot_i4_wave`),
/// zero, negative, or the `amax == 0` sentinel `1.0` (a dead group). Cheap: scales are
/// ~2 MB per expert at `I4_GROUP` = 128, so this preads a few hundred MB/layer.
fn scan_scales(art: &str, cfg: &ModelConfig) -> Result<()> {
    let (h, m, ne) = (cfg.hidden, cfg.moe_inter, cfg.n_experts);
    let off = i4_slot_offsets(h, m);
    let stride = i4_expert_stride(h, m);
    let dims = vq_expert_layout(h, m);
    let (mut nonfinite, mut zero, mut neg, mut ones, mut total) = (0u64, 0u64, 0u64, 0u64, 0u64);
    let (mut lo, mut hi) = (f32::INFINITY, 0f32);
    let mut worst = (0usize, 0usize, 0usize, 0usize, 0f32); // layer, expert, proj, slot, scale
    for l in cfg.dense_layers..cfg.n_layers {
        let f = File::open(format!("{art}/L{l:02}.i4"))?;
        for e in 0..=ne {
            for (k, &(o_dim, i_dim)) in dims.iter().enumerate() {
                let mut b = vec![0u8; o_dim * i4_groups(i_dim) * 4];
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
        "scales: {total} total (group {I4_GROUP}), nonfinite {nonfinite}, zero {zero}, negative {neg}, ==1.0 (amax==0 dead group) {ones}"
    );
    println!(
        "        min {lo:.6e}  max {hi:.6e}  (max at layer {} expert {} proj {} slot {}, s={:.6e})",
        worst.0, worst.1, worst.2, worst.3, worst.4
    );
    println!(
        "        max|W| implied by the largest scale = 7·s = {:.4e}",
        hi * 7.0
    );
    Ok(())
}

/// Warn if the artifact's `i4_source` stamp does not name the `fp8` directory being
/// audited, or is absent.
///
/// The checkpoint is taken on faith otherwise: a DIFFERENT revision of the same model
/// passes `dequant_fp8`'s shape checks, turns every byte-identity check false, and makes
/// the report read "the shipped .i4 is defective" when the ARGUMENT is.
fn warn_provenance(art: &str, fp8: &str) -> Result<()> {
    match I4Source::load(art)? {
        Some(p)
            if std::fs::canonicalize(fp8)
                .map(|c| c.display().to_string())
                .map(|c| c != p.src)
                .unwrap_or(true) =>
        {
            eprintln!(
                "WARNING: artifact was built from {} — auditing against {fp8}",
                p.src
            )
        }
        None => eprintln!("WARNING: artifact carries no i4_source stamp; provenance unverified"),
        _ => {}
    }
    Ok(())
}

/// One expert's `.vq3` block, at its `VQ_ALIGN`-headered offset.
///
/// Both readers here need the same two facts — the header offset and the stride — and a
/// second copy of that arithmetic is a second chance to read one expert's bytes as
/// another's, which decodes to plausible garbage rather than failing.
fn read_vq_block(vqf: &File, e: usize, vq_stride: usize) -> Result<Vec<u8>> {
    let mut vqb = vec![0u8; vq_stride];
    vqf.read_exact_at(
        &mut vqb,
        (rivoli::artifact::quant::VQ_ALIGN + e * vq_stride) as u64,
    )
    .with_context(|| format!("read .vq3 expert {e}"))?;
    Ok(vqb)
}

fn main() -> Result<()> {
    // Destructured, not held: the mode flags below are read in a fixed precedence
    // (`--scan`, then `--xcheck`, `--verify`, `--scale-study`, then the default report),
    // and naming them here is what makes that ladder readable at the dispatch site.
    let Args {
        artifact_dir: art,
        fp8_dir: fp8,
        scan,
        verify,
        xcheck: xchk,
        scale_study: study,
        layer,
        experts,
    } = Args::parse();

    let cfg = ModelConfig::load(&art)?;
    if scan {
        return scan_scales(&art, &cfg);
    }
    // Provenance BEFORE the mode dispatch: `verify_wide` is a byte-identity check against
    // the operator-supplied fp8 dir, so it is the mode where auditing against a different
    // revision of the same model is most damaging — every check goes false and the report
    // blames the artifact for the argument. (`--xcheck` reads no fp8 and is unaffected, but
    // a stale stamp is worth surfacing there too.)
    warn_provenance(&art, &fp8)?;
    if xchk {
        let ls: Vec<usize> = (cfg.dense_layers..cfg.n_layers).step_by(3).collect();
        let es: Vec<usize> = (0..cfg.n_experts)
            .step_by(17)
            .chain([cfg.n_experts])
            .collect();
        return xcheck(&art, &cfg, &ls, &es);
    }
    if verify {
        let ls: Vec<usize> = (cfg.dense_layers..cfg.n_layers).step_by(6).collect();
        let es: Vec<usize> = (0..cfg.n_experts)
            .step_by(31)
            .chain([cfg.n_experts])
            .collect();
        return verify_wide(&art, &fp8, &cfg, &ls, &es);
    }
    // Reads only the fp8 checkpoint, so it is valid against an artifact of any
    // generation — including before a rebuild, which is the point: it is what decides
    // whether a rebuild is worth buying.
    if study {
        let block = FormatMeta::load(&art)?.fp8_block;
        return scale_study_only(&fp8, &cfg, block, layer, &experts);
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
    // REDUNDANT in this mode — the pre-dispatch call above has already run and printed
    // the same line. Kept because removing it changes what the tool prints, which this
    // pass is not allowed to do; drop it (and this comment) in a change that is.
    warn_provenance(&art, &fp8)?;
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
        let vqb = read_vq_block(&vqf, e, vq_stride)?;
        let vqp = vq_expert(&vqb, 0, h, m);

        let base = expert_base(layer, e, ne);

        // Kept per projection so the whole-expert chain below can be run both ways.
        let (mut wrefs, mut i4s, mut vqs) = (vec![], vec![], vec![]);
        for (k, (&(o_dim, i_dim), pname)) in dims.iter().zip(PROJ).enumerate() {
            // ── independent ground truth: the fp8 checkpoint ─────────────────
            let wref = src.dequant_fp8(&format!("{base}.{pname}"), o_dim, i_dim, block)?;

            // ── what the artifact actually holds ─────────────────────────────
            let rb = i4_row_bytes(i_dim);
            let packed = &i4b[off[k * 2]..off[k * 2] + o_dim * rb];
            let scale =
                read_f32(&i4b[off[k * 2 + 1]..off[k * 2 + 1] + o_dim * i4_groups(i_dim) * 4]);

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
                e,
                pname,
                aw.rel_l2,
                aw.max_abs,
                aw.cos,
                aw.gain,
                ay.rel_l2,
                ay.cos,
                ay.gain,
                if bytes_match && scales_match {
                    ""
                } else {
                    "  <<< DISK != quant_i4(fp8)"
                }
            );

            // ── the `.vq3` at the same coordinates: `--mode int3-vq`'s own arithmetic,
            //    scored against the same fp8 truth. Both shipped formats derive from this
            //    checkpoint by different quantizers, so this is the standing head-to-head
            //    — and it is decoded, not re-quantized, so it is what that mode computes.
            let wvq = vq_decode_proj(&vqp[k], &cbs[k]);
            let avq = agree(&wvq, &wref);
            let yvq = matvec_f64(&wvq, &x, o_dim, i_dim);
            let ayvq = agree(&yvq, &yref);
            println!(
                "{:<6} {:<10} | {:>10.4e} {:>10.3e} {:>9.6} {:>9.6} | {:>10.4e} {:>9.6} {:>9.6} |",
                "",
                "  ^vq3",
                avq.rel_l2,
                avq.max_abs,
                avq.cos,
                avq.gain,
                ayvq.rel_l2,
                ayvq.cos,
                ayvq.gain
            );
            // ── DEAD ROWS. A per-ROW amax/7 step means one outlier sets the scale for
            //    all i_dim weights; every weight below s/2 then rounds to nibble 8 = 0.
            //    Group scales cannot fail that way — an outlier can only flatten its own
            //    group — and this is the statistic that priced the move: 603 of 6144 rows
            //    of L03 e0 down_proj were past 50% zeros under per-row scales
            //    (docs/investigations/int4-scales.md §6, the acceptance gate). A per-row
            //    error metric averages it away: a 100%-zero row contributes only its own
            //    share of rel-L2 while removing an output channel outright.
            let mut zero_frac: Vec<f64> = packed
                .chunks_exact(rb)
                .map(|row| {
                    let z = (0..i_dim)
                        .filter(|&i| {
                            let b = row[i >> 1];
                            (if i & 1 == 0 { b & 0x0F } else { b >> 4 }) == 8
                        })
                        .count();
                    z as f64 / i_dim as f64
                })
                .collect();
            let dead_mean = zero_frac.iter().sum::<f64>() / o_dim as f64;
            zero_frac.sort_by(f64::total_cmp);
            println!(
                "{:<6} DEAD {:<8}| mean {:.3}  p99 {:.3}  max {:.3}  >50% {:>4}  >80% {:>4}  ==100% {:>4}",
                e,
                pname,
                dead_mean,
                zero_frac[o_dim * 99 / 100],
                zero_frac[o_dim - 1],
                zero_frac.iter().filter(|&&f| f > 0.5).count(),
                zero_frac.iter().filter(|&&f| f > 0.8).count(),
                zero_frac.iter().filter(|&&f| f >= 1.0).count()
            );

            // ── bulk vs tail: whole-row rel-L2 is dominated by the large weights and can
            //    improve while the many small ones — where decode quality lives — get
            //    coarser. The split is taken on the fp8 row's own p99, so it prices the
            //    bulk without the tail's leverage.
            let b = bulk_stats(&wref, &wi4, packed, &scale, o_dim, i_dim);
            println!(
                "{:<6} BULK {:<8}| amax/med {:>5.1}  amax/p99.9 {:>5.2}  spiky rows {:>5}/{o_dim}  \
                 dead rows {:>5}/{o_dim}  step {:>9.3e}  bulk relL2 {:>7.4}  tail relL2 {:>7.4}  \
                 levels {:>2}/16",
                e,
                pname,
                b.amax_over_med,
                b.amax_over_p999,
                b.spiky_rows,
                b.dead_rows,
                b.step_mean,
                b.bulk_rel_l2,
                b.tail_rel_l2,
                b.levels_used
            );
            println!("{:<6} HIST {:<8}| {:?}", e, pname, b.hist);
            wrefs.push(wref);
            i4s.push((packed.to_vec(), scale));
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
        // The one clone in this loop, and it is here because `chain_i4` wants the three
        // projections as an owned array while `i4s` has to outlive it — `packed` borrows
        // the layer buffer, so nothing here can hand out a `&[(&[u8], &[f32])]` that also
        // survives the projection loop above.
        let newq: [(Vec<u8>, Vec<f32>); 3] = core::array::from_fn(|k| i4s[k].clone());
        let cn = agree(&chain_i4(&newq), &chain_ref);
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
        for (label, a) in [("i4", cn), ("vq3", cv)] {
            println!(
                "{:<6} CHAIN {:<7}| relL2 {:>7.4}  gain {:>7.4}  cos {:>8.6}  maxerr/max|ref| {:>7.4}",
                e,
                label,
                a.rel_l2,
                a.gain,
                a.cos,
                a.max_abs / ymax
            );
        }
    }
    println!(
        "\nBYTE IDENTITY: {} of {} projections differ from quant_i4(dequant_fp8(ckpt))",
        mismatches,
        experts.len() * 3
    );
    ensure!(
        mismatches == 0,
        "the shipped .i4 is NOT what this checkpoint quantizes to"
    );
    Ok(())
}
