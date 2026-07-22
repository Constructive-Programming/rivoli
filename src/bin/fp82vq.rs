//! fp82vq — offline converter: GLM-5.2 fp8 experts → group-scaled VQ-int3 (M2, see
//! `docs/int3.md`). Reads the fp8 checkpoint shards (F8_E4M3 weights + F32 128×128
//! block scales), learns a `VQ_DIM`-d codebook by k-means over a sample of experts,
//! then writes ONE file per MoE layer: 256 experts at a fixed stride, each laid out
//! `gate ‖ up ‖ down`, each projection = packed 12-bit codebook indices then bf16
//! group scales. The learned codebook is written once to `<out>/codebook.f32`.
//!
//! Pure CPU — build and run without the GPU toolchain. The pass-2 encode is the
//! expensive part (nearest-of-VQ_K per subvector over all experts); it parallelizes
//! across experts with `std::thread::scope`.
//!
//! usage: fp82vq <fp8-dir> <out-dir> [--sample-experts N] [--kmeans-iters N]
//!
//! ponytail: pass-2 encode is brute-force nearest over VQ_K centroids — O(K·d) per
//! subvector, the run-time ceiling. If a full conversion is too slow, the upgrade is
//! a lattice codebook (analytic O(d) nearest, the reason E8/IQ3 use one) or a k-d
//! tree over the learned centroids; the on-disk format and the kernel are unchanged.

use anyhow::{Context, Result, ensure};
use memmap2::Mmap;
use rivoli::math::{bf16_to_f32, f32_to_bf16};
use rivoli::quant::{
    VQ_DIM, VQ_GROUP, VQ_K, dequant_fp8_block, matvec_vq, quant_vq, vq_expert_stride, vq_groups,
    vq_proj_bytes, vq_row_bytes,
};
use std::collections::HashMap;
use std::io::Write;

const FP8_BLOCK: usize = 128; // GLM-5.2 fp8 weight_scale_inv tile (128×128)

/// One tensor's location in the mmap'd fp8 checkpoint.
struct Loc {
    shard: usize,
    begin: usize,
    len: usize,
    shape: Vec<usize>,
}

/// The fp8 checkpoint: every `model-*.safetensors` shard mmap'd, tensors indexed.
struct Fp8Src {
    mmaps: Vec<Mmap>,
    index: HashMap<String, Loc>,
}

impl Fp8Src {
    fn open(dir: &str) -> Result<Self> {
        let mut paths: Vec<_> = std::fs::read_dir(dir)
            .with_context(|| format!("read fp8 dir {dir}"))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("model-") && n.ends_with(".safetensors"))
            })
            .collect();
        paths.sort();
        ensure!(!paths.is_empty(), "no model-*.safetensors in {dir}");
        let mut mmaps = Vec::with_capacity(paths.len());
        let mut index = HashMap::new();
        for path in &paths {
            let file = std::fs::File::open(path).with_context(|| format!("open {path:?}"))?;
            // SAFETY: the checkpoint shards are read-only for the converter's lifetime.
            let mmap = unsafe { Mmap::map(&file) }.with_context(|| format!("mmap {path:?}"))?;
            let hlen = u64::from_le_bytes(mmap[..8].try_into()?) as usize;
            let hdr: serde_json::Value = serde_json::from_slice(&mmap[8..8 + hlen])
                .with_context(|| format!("parse header {path:?}"))?;
            let base = 8 + hlen;
            let mut entries = Vec::new();
            for (name, t) in hdr
                .as_object()
                .context("safetensors header not an object")?
            {
                if name == "__metadata__" {
                    continue;
                }
                let off = t["data_offsets"].as_array().context("data_offsets")?;
                let u = |v: &serde_json::Value| v.as_u64().context("non-integer header field");
                let (b, e) = (u(&off[0])? as usize, u(&off[1])? as usize);
                let shape = t["shape"]
                    .as_array()
                    .context("shape")?
                    .iter()
                    .map(|v| u(v).map(|x| x as usize))
                    .collect::<Result<_>>()?;
                entries.push((
                    name.clone(),
                    Loc {
                        shard: mmaps.len(), // position of THIS mmap once pushed (skips shift it)
                        begin: base + b,
                        len: e - b,
                        shape,
                    },
                ));
            }
            // A shard still downloading is shorter than its header promises — skip it
            // whole (its tensors read as absent) rather than panic slicing past EOF.
            let need = entries
                .iter()
                .map(|(_, l)| l.begin + l.len)
                .max()
                .unwrap_or(0);
            if mmap.len() < need {
                eprintln!(
                    "fp82vq: {path:?} incomplete ({} < {need} B), skipping",
                    mmap.len()
                );
                continue;
            }
            for (name, loc) in entries {
                index.insert(name, loc);
            }
            mmaps.push(mmap);
        }
        Ok(Self { mmaps, index })
    }

    fn loc(&self, name: &str) -> Result<&Loc> {
        self.index
            .get(name)
            .with_context(|| format!("tensor {name} not in fp8 checkpoint"))
    }

    fn bytes(&self, name: &str) -> Result<&[u8]> {
        let l = self.loc(name)?;
        Ok(&self.mmaps[l.shard][l.begin..l.begin + l.len])
    }
}

/// Dequantize one expert projection `<base>.<proj>.weight` (F8_E4M3) with its
/// `weight_scale_inv` (F32 128×128 blocks) to f32 `[o_dim·i_dim]`. `o_dim`/`i_dim`
/// come from the tensor's own shape (checked against the caller's expectation).
fn deq_proj(src: &Fp8Src, base: &str, proj: &str, o_dim: usize, i_dim: usize) -> Result<Vec<f32>> {
    let wname = format!("{base}.{proj}.weight");
    let shape = &src.loc(&wname)?.shape;
    ensure!(
        shape.as_slice() == [o_dim, i_dim],
        "{wname}: shape {shape:?}, expected [{o_dim}, {i_dim}]"
    );
    let w = src.bytes(&wname)?;
    let scale = rivoli::quant::read_f32(src.bytes(&format!("{base}.{proj}.weight_scale_inv"))?);
    Ok(dequant_fp8_block(w, &scale, o_dim, i_dim, FP8_BLOCK))
}

/// Append every `stride`-th group-normalized `VQ_DIM`-subvector of `w[o_dim,i_dim]`
/// to `out` — the same normalization `quant_vq` applies, so the codebook is learned
/// on exactly the distribution it will encode. `stride` keeps the sample bounded (a
/// full expert alone is ~9.4M subvectors).
///
/// ponytail: subvectors count equally. s²-weighting (GEMV-error-proportional) was
/// tried and net-regressed — it tilted the one shared codebook toward gate/up
/// (6144-wide) and starved down_proj (2048-wide). The lever that data points at is a
/// per-projection codebook, not weighting; deferred until E2E perplexity asks for it.
fn push_normalized_subvectors(
    w: &[f32],
    o_dim: usize,
    i_dim: usize,
    stride: usize,
    out: &mut Vec<f32>,
) {
    let mut n = 0usize;
    for o in 0..o_dim {
        let row = &w[o * i_dim..(o + 1) * i_dim];
        for grp in row.chunks_exact(VQ_GROUP) {
            let amax = grp.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            let inv = 1.0 / bf16_to_f32(f32_to_bf16(if amax > 0.0 { amax } else { 1.0 }));
            for sub in grp.chunks_exact(VQ_DIM) {
                if n.is_multiple_of(stride) {
                    out.extend(sub.iter().map(|&v| v * inv));
                }
                n += 1;
            }
        }
    }
}

/// k-means for the codebook: k-means++ seeding (deterministic xorshift), threaded
/// Lloyd, stops early when distortion improves by < 0.01% (`max_iters` caps it).
fn learn_codebook(sample: &[f32], max_iters: usize) -> Vec<f32> {
    let n = sample.len() / VQ_DIM;
    assert!(n >= VQ_K, "sample {n} smaller than VQ_K {VQ_K}");
    let threads = std::thread::available_parallelism().map_or(4, |t| t.get());
    let mut c = vec![0.0f32; VQ_K * VQ_DIM];

    // k-means++ : each next seed drawn ∝ D²(point, nearest seed so far).
    let mut rng = 0x2545_F491_4F6C_DD1Du64;
    let mut next_f64 = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut mind = vec![1.0f32; n]; // uniform draw for the first seed
    for j in 0..VQ_K {
        let total: f64 = mind.iter().map(|&d| d as f64).sum();
        let mut target = next_f64() * total;
        let mut pick = n - 1;
        for (i, &d) in mind.iter().enumerate() {
            target -= d as f64;
            if target <= 0.0 {
                pick = i;
                break;
            }
        }
        let seed = &sample[pick * VQ_DIM..(pick + 1) * VQ_DIM];
        c[j * VQ_DIM..(j + 1) * VQ_DIM].copy_from_slice(seed);
        for (i, md) in mind.iter_mut().enumerate() {
            let v = &sample[i * VQ_DIM..(i + 1) * VQ_DIM];
            let d: f32 = v.iter().zip(seed).map(|(&x, &y)| (x - y) * (x - y)).sum();
            *md = if j == 0 { d } else { md.min(d) };
        }
    }

    // Lloyd: threaded assignment accumulating sums per centroid.
    let mut prev = f64::INFINITY;
    for it in 0..max_iters {
        let chunk = n.div_ceil(threads);
        let parts: Vec<(Vec<f32>, Vec<u32>, f64)> = std::thread::scope(|s| {
            let mut handles = Vec::new();
            for t in 0..threads {
                let (lo, hi) = (t * chunk, ((t + 1) * chunk).min(n));
                if lo >= hi {
                    break;
                }
                let c = &c;
                handles.push(s.spawn(move || {
                    let mut sum = vec![0.0f32; VQ_K * VQ_DIM];
                    let mut cnt = vec![0u32; VQ_K];
                    let mut dist = 0.0f64;
                    for i in lo..hi {
                        let v = &sample[i * VQ_DIM..(i + 1) * VQ_DIM];
                        let mut best = (f32::INFINITY, 0usize);
                        for j in 0..VQ_K {
                            let cc = &c[j * VQ_DIM..(j + 1) * VQ_DIM];
                            let d: f32 = v.iter().zip(cc).map(|(&x, &y)| (x - y) * (x - y)).sum();
                            if d < best.0 {
                                best = (d, j);
                            }
                        }
                        dist += best.0 as f64;
                        cnt[best.1] += 1;
                        for d in 0..VQ_DIM {
                            sum[best.1 * VQ_DIM + d] += v[d];
                        }
                    }
                    (sum, cnt, dist)
                }));
            }
            handles.into_iter().filter_map(|h| h.join().ok()).collect()
        });
        let mut sum = vec![0.0f32; VQ_K * VQ_DIM];
        let mut cnt = vec![0u32; VQ_K];
        let mut dist = 0.0f64;
        for (ps, pc, pd) in parts {
            for (a, b) in sum.iter_mut().zip(ps) {
                *a += b;
            }
            for (a, b) in cnt.iter_mut().zip(pc) {
                *a += b;
            }
            dist += pd;
        }
        for j in 0..VQ_K {
            if cnt[j] > 0 {
                for d in 0..VQ_DIM {
                    c[j * VQ_DIM + d] = sum[j * VQ_DIM + d] / cnt[j] as f32;
                }
            } // empty cluster: keep its k-means++ seed
        }
        let improved = (prev - dist) / dist.max(f64::MIN_POSITIVE);
        if improved.abs() < 1e-4 {
            eprintln!("fp82vq: k-means converged after {} iters", it + 1);
            break;
        }
        prev = dist;
    }
    c
}

/// The three projections of a GLM-5.2 expert, in on-disk order: (name, o_dim, i_dim).
fn projections(hidden: usize, moe_inter: usize) -> [(&'static str, usize, usize); 3] {
    [
        ("gate_proj", moe_inter, hidden),
        ("up_proj", moe_inter, hidden),
        ("down_proj", hidden, moe_inter),
    ]
}

/// Write one projection's `(indices, scales)` into `dst` (indices first, then bf16
/// scales as little-endian u16) — the layout [`vq_proj_bytes`] sizes.
fn write_proj(dst: &mut [u8], o_dim: usize, i_dim: usize, indices: &[u8], scales: &[u16]) {
    let idx_bytes = o_dim * vq_row_bytes(i_dim);
    debug_assert_eq!(indices.len(), idx_bytes);
    debug_assert_eq!(scales.len(), o_dim * vq_groups(i_dim));
    dst[..idx_bytes].copy_from_slice(indices);
    for (s, out) in scales.iter().zip(dst[idx_bytes..].chunks_exact_mut(2)) {
        out.copy_from_slice(&s.to_le_bytes());
    }
}

/// Quantize one expert into `dst` (length `expert_stride`), against `codebook`.
fn convert_expert(
    src: &Fp8Src,
    layer: usize,
    e: usize,
    hidden: usize,
    moe_inter: usize,
    codebook: &[f32],
    dst: &mut [u8],
) -> Result<()> {
    let base = format!("model.layers.{layer}.mlp.experts.{e}");
    let mut off = 0;
    for (proj, o_dim, i_dim) in projections(hidden, moe_inter) {
        let w = deq_proj(src, &base, proj, o_dim, i_dim)?;
        let (indices, scales) = quant_vq(&w, o_dim, i_dim, codebook);
        let pb = vq_proj_bytes(o_dim, i_dim);
        write_proj(&mut dst[off..off + pb], o_dim, i_dim, &indices, &scales);
        off += pb;
    }
    Ok(())
}

/// `--check`: GEMV relative RMS of the on-disk VQ bytes vs the fp8 oracle, with a
/// per-row int4 (colibri-style, s=amax/7) baseline on the same weights — the M2
/// acceptance gate: VQ-int3 must ≈ int4.
fn check_layer(
    src: &Fp8Src,
    out_dir: &str,
    l: usize,
    hidden: usize,
    moe_inter: usize,
    stride: usize,
) -> Result<()> {
    let codebook = rivoli::quant::read_f32(&std::fs::read(format!("{out_dir}/codebook.f32"))?);
    ensure!(codebook.len() == VQ_K * VQ_DIM, "bad codebook size");
    let f = std::fs::File::open(format!("{out_dir}/L{l:02}.i3"))?;
    // SAFETY: read-only for the check's lifetime.
    let mm = unsafe { Mmap::map(&f)? };
    for e in [0usize, 128, 255] {
        let blk = &mm[e * stride..(e + 1) * stride];
        let mut off = 0;
        let base = format!("model.layers.{l}.mlp.experts.{e}");
        for (proj, o_dim, i_dim) in projections(hidden, moe_inter) {
            let w = deq_proj(src, &base, proj, o_dim, i_dim)?;
            let idx_bytes = o_dim * vq_row_bytes(i_dim);
            let indices = &blk[off..off + idx_bytes];
            let scales: Vec<u16> = blk[off + idx_bytes..off + vq_proj_bytes(o_dim, i_dim)]
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            off += vq_proj_bytes(o_dim, i_dim);
            // Deterministic pseudo-random input, ~U[-1,1].
            let mut s = 0x9E3779B97F4A7C15u64.wrapping_add(e as u64);
            let x: Vec<f32> = (0..i_dim)
                .map(|_| {
                    s = s
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
                })
                .collect();
            let mut y_ref = vec![0.0f32; o_dim];
            for (o, yo) in y_ref.iter_mut().enumerate() {
                *yo = w[o * i_dim..(o + 1) * i_dim]
                    .iter()
                    .zip(&x)
                    .map(|(&a, &b)| a * b)
                    .sum();
            }
            let mut y_vq = vec![0.0f32; o_dim];
            matvec_vq(&mut y_vq, &x, indices, &scales, &codebook, o_dim, i_dim);
            let mut y_i4 = vec![0.0f32; o_dim];
            for (o, yo) in y_i4.iter_mut().enumerate() {
                let row = &w[o * i_dim..(o + 1) * i_dim];
                let amax = row.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
                let sc = if amax > 0.0 { amax / 7.0 } else { 1.0 };
                *yo = row
                    .iter()
                    .zip(&x)
                    .map(|(&a, &b)| (a / sc).round().clamp(-8.0, 7.0) * sc * b)
                    .sum();
            }
            let rel = |y: &[f32]| -> f32 {
                let num: f32 = y.iter().zip(&y_ref).map(|(&a, &b)| (a - b) * (a - b)).sum();
                let den: f32 = y_ref.iter().map(|&b| b * b).sum();
                (num / den).sqrt()
            };
            println!(
                "L{l} e{e:3} {proj:9}: vq3 {:.4} | int4 {:.4}",
                rel(&y_vq),
                rel(&y_i4)
            );
        }
    }
    Ok(())
}

struct Args {
    fp8_dir: String,
    out_dir: String,
    sample_experts: usize,
    kmeans_iters: usize,
    /// Convert only this layer (smoke tests / partial checkpoints); codebook is then
    /// sampled from this layer's experts instead of across layers.
    layer: Option<usize>,
    /// Validate an already-converted layer instead of converting: GEMV relative RMS
    /// of the on-disk VQ bytes vs the fp8 oracle, with a per-row int4 baseline.
    check: bool,
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        fp8_dir: String::new(),
        out_dir: String::new(),
        sample_experts: 64,
        kmeans_iters: 15,
        layer: None,
        check: false,
    };
    let mut pos = Vec::new();
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--sample-experts" => {
                a.sample_experts = it.next().context("--sample-experts N")?.parse()?;
            }
            "--kmeans-iters" => {
                a.kmeans_iters = it.next().context("--kmeans-iters N")?.parse()?;
            }
            "--layer" => {
                a.layer = Some(it.next().context("--layer N")?.parse()?);
            }
            "--check" => a.check = true,
            _ => pos.push(arg),
        }
    }
    ensure!(
        pos.len() == 2,
        "usage: fp82vq <fp8-dir> <out-dir> [--sample-experts N] [--kmeans-iters N] [--layer N] [--check]"
    );
    a.fp8_dir = pos[0].clone();
    a.out_dir = pos[1].clone();
    Ok(a)
}

fn main() -> Result<()> {
    let args = parse_args()?;
    // Model dims from the fp8 checkpoint's config.json.
    let cfg: serde_json::Value =
        serde_json::from_slice(&std::fs::read(format!("{}/config.json", args.fp8_dir))?)?;
    let g = |k: &str| -> Result<usize> {
        Ok(cfg[k]
            .as_u64()
            .with_context(|| format!("config missing {k}"))? as usize)
    };
    let (hidden, moe_inter) = (g("hidden_size")?, g("moe_intermediate_size")?);
    let n_experts = g("n_routed_experts").or_else(|_| g("num_experts"))?;
    let n_layers = g("num_hidden_layers")?;
    let dense = g("first_k_dense_replace")?;
    for (d, name) in [(hidden, "hidden"), (moe_inter, "moe_inter")] {
        ensure!(
            d % VQ_GROUP == 0 && d % VQ_DIM == 0,
            "{name} {d} not divisible by VQ params"
        );
    }
    let stride = vq_expert_stride(hidden, moe_inter);
    eprintln!(
        "fp82vq: hidden={hidden} moe_inter={moe_inter} experts={n_experts} layers={dense}..{n_layers} \
         | VQ_DIM={VQ_DIM} VQ_K={VQ_K} VQ_GROUP={VQ_GROUP} | expert {stride} B, layer {} GiB",
        (n_experts * stride) as f64 / (1u64 << 30) as f64
    );
    std::fs::create_dir_all(&args.out_dir)?;
    let src = Fp8Src::open(&args.fp8_dir)?;

    if args.check {
        let l = args.layer.context("--check requires --layer")?;
        return check_layer(&src, &args.out_dir, l, hidden, moe_inter, stride);
    }

    // Pass 1 — learn the codebook from a sample of experts: spread across MoE layers
    // normally, or across this one layer's experts under --layer. Subvector stride
    // caps the k-means sample at ~TARGET regardless of how many experts are sampled.
    const TARGET_SAMPLE: usize = 1 << 20;
    let pairs: Vec<(usize, usize)> = match args.layer {
        Some(l) => (0..n_experts)
            .step_by((n_experts / args.sample_experts).max(1))
            .map(|e| (l, e))
            .collect(),
        None => (dense..n_layers)
            .step_by(((n_layers - dense) / args.sample_experts).max(1))
            .map(|l| (l, l % n_experts)) // vary the expert too, cheap spread
            .collect(),
    };
    // A codebook already in out_dir is REUSED (idempotent incremental runs must
    // encode every layer against the same codebook); otherwise learn one from
    // whatever sampled experts are present (partial checkpoints skip, with a warn)
    // and persist it.
    let cb_path = format!("{}/codebook.f32", args.out_dir);
    let codebook = if let Ok(b) = std::fs::read(&cb_path) {
        eprintln!("fp82vq: reusing {cb_path}");
        let cb = rivoli::quant::read_f32(&b);
        ensure!(cb.len() == VQ_K * VQ_DIM, "bad codebook size in {cb_path}");
        cb
    } else {
        let per_expert = (2 * moe_inter * hidden + hidden * moe_inter) / VQ_DIM;
        let stride_s = (pairs.len() * per_expert / TARGET_SAMPLE).max(1);
        let mut sample = Vec::new();
        let mut skipped = 0;
        for &(l, e) in &pairs {
            for (proj, o_dim, i_dim) in projections(hidden, moe_inter) {
                match deq_proj(
                    &src,
                    &format!("model.layers.{l}.mlp.experts.{e}"),
                    proj,
                    o_dim,
                    i_dim,
                ) {
                    Ok(w) => push_normalized_subvectors(&w, o_dim, i_dim, stride_s, &mut sample),
                    Err(_) => skipped += 1,
                }
            }
        }
        eprintln!(
            "fp82vq: learning codebook from {} subvectors ({skipped} sampled tensors absent)…",
            sample.len() / VQ_DIM
        );
        let codebook = learn_codebook(&sample, args.kmeans_iters);
        let mut cb_bytes = Vec::with_capacity(codebook.len() * 4);
        for &v in &codebook {
            cb_bytes.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(&cb_path, &cb_bytes)?;
        codebook
    };

    // Pass 2 — convert the selected layers, experts in parallel. Skips layers whose
    // output already exists and layers whose fp8 tensors haven't downloaded yet, so
    // re-running converges on a partial checkpoint as shards land.
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let layers = args.layer.map_or(dense..n_layers, |l| l..l + 1);
    for l in layers {
        let path = format!("{}/L{l:02}.i3", args.out_dir);
        if std::fs::metadata(&path).is_ok() {
            eprintln!("fp82vq: {path} exists, skipping");
            continue;
        }
        if src
            .loc(&format!("model.layers.{l}.mlp.experts.0.gate_proj.weight"))
            .is_err()
        {
            eprintln!("fp82vq: layer {l} tensors not present, skipping");
            continue;
        }
        let mut buf = vec![0u8; n_experts * stride];
        let src = &src;
        let codebook = &codebook;
        // Hand each thread a disjoint index range and its own slice of `buf`; each
        // returns a Result, so an error propagates without a shared mutex.
        let per = n_experts.div_ceil(threads);
        std::thread::scope(|s| -> Result<()> {
            let mut handles = Vec::new();
            let mut rest = &mut buf[..];
            let mut e0 = 0;
            while e0 < n_experts {
                let take = per.min(n_experts - e0);
                let (mine, tail) = rest.split_at_mut(take * stride);
                rest = tail;
                let base_e = e0;
                e0 += take;
                handles.push(s.spawn(move || -> Result<()> {
                    for (j, dst) in mine.chunks_exact_mut(stride).enumerate() {
                        convert_expert(src, l, base_e + j, hidden, moe_inter, codebook, dst)?;
                    }
                    Ok(())
                }));
            }
            for h in handles {
                h.join()
                    .map_err(|_| anyhow::anyhow!("converter worker panicked"))??;
            }
            Ok(())
        })
        .with_context(|| format!("convert layer {l}"))?;
        let mut f = std::fs::File::create(&path)?;
        f.write_all(&buf)?;
        eprintln!("fp82vq: wrote {path} ({} experts)", n_experts);
    }
    eprintln!(
        "fp82vq: done — {} layers to {}",
        n_layers - dense,
        args.out_dir
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rivoli::quant::matvec_vq;

    #[test]
    fn kmeans_recovers_separated_clusters() {
        // Two tight clusters far apart; every centroid should sit on one of them.
        let mut sample = Vec::new();
        for i in 0..(VQ_K * 4) {
            let c = if i % 2 == 0 { -10.0 } else { 10.0 };
            let jitter = ((i as f32 * 7.0).sin()) * 0.01;
            for _ in 0..VQ_DIM {
                sample.push(c + jitter);
            }
        }
        let cb = learn_codebook(&sample, 5);
        for j in 0..VQ_K {
            let m = cb[j * VQ_DIM]; // first coord
            assert!(
                (m + 10.0).abs() < 0.5 || (m - 10.0).abs() < 0.5,
                "centroid {j} = {m}"
            );
        }
    }

    #[test]
    fn expert_layout_roundtrips_through_disk_bytes() {
        // One projection's (indices, scales) written to the on-disk layout must read
        // back so matvec_vq reproduces the quantized GEMV — the converter↔loader contract.
        let (o_dim, i_dim) = (8usize, VQ_GROUP); // small but valid (i_dim = one group)
        // A codebook that contains the two subvectors we'll use, unit-peak.
        let mut cb = vec![1e30f32; VQ_K * VQ_DIM];
        cb[0..VQ_DIM].copy_from_slice(&[1.0, -1.0, 0.5, -0.5]);
        cb[VQ_DIM..2 * VQ_DIM].copy_from_slice(&[0.25, 0.5, 0.75, 1.0]);
        let mut w = vec![0.0f32; o_dim * i_dim];
        for (t, chunk) in w.chunks_exact_mut(VQ_DIM).enumerate() {
            let e = if t % 2 == 0 { 0 } else { 1 };
            for d in 0..VQ_DIM {
                chunk[d] = cb[e * VQ_DIM + d] * 0.5;
            }
        }
        let (indices, scales) = quant_vq(&w, o_dim, i_dim, &cb);
        // write to disk layout, read the two arrays back
        let pb = vq_proj_bytes(o_dim, i_dim);
        let mut disk = vec![0u8; pb];
        write_proj(&mut disk, o_dim, i_dim, &indices, &scales);
        let idx_bytes = o_dim * vq_row_bytes(i_dim);
        let (r_idx, r_scale_bytes) = disk.split_at(idx_bytes);
        let r_scales: Vec<u16> = r_scale_bytes
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        // GEMV via the read-back arrays must equal GEMV via the originals.
        let x: Vec<f32> = (0..i_dim).map(|k| (k + 1) as f32).collect();
        let (mut y0, mut y1) = (vec![0.0f32; o_dim], vec![0.0f32; o_dim]);
        matvec_vq(&mut y0, &x, &indices, &scales, &cb, o_dim, i_dim);
        matvec_vq(&mut y1, &x, r_idx, &r_scales, &cb, o_dim, i_dim);
        assert_eq!(y0, y1);
    }
}
