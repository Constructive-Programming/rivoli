//! convert — GLM-5.2 fp8 checkpoint → the rivoli int3-vq artifact (see
//! docs/reference/architecture.md + src/artifact/format.rs). Learns 3 per-projection codebooks,
//! VQ-encodes the routed + shared experts into per-layer `.vq3` files, and
//! assembles the resident set (attention/dense fp8 copied native, norms + router
//! gate bf16→f32, embed/lm_head bf16→int8) into `resident.safetensors`, plus a
//! self-describing `manifest.json`.
//!
//! CPU encode by default (correct, slow); `--layers N` limits the layer count for
//! quick artifact/loader validation. `--help` is the flag reference.

use anyhow::{Context, Result, ensure};
use clap::Parser;
use rivoli::artifact::format::{Dtype, FormatMeta, SafeWriter, Safetensors, Vq3Header};
use rivoli::math::bf16_to_f32;
use rivoli::artifact::quant::{
    ExpertProjs, PROJ, VQ_ALIGN, VQ_DIM, VQ_K, expert_base, expert_projs, learn_codebook, quant_vq,
    read_f32, sample_subvectors, vq_expert_bytes, vq_proj_bytes, vq_row_bytes, write_le_scales,
};
// Only the tests still take the layout on its own; the encoders go through `expert_projs`,
// which carries the projection NAMES alongside it. Gated rather than allow-ed, for the
// same reason as `VQ_GROUP` below.
#[cfg(test)]
use rivoli::artifact::quant::vq_expert_layout;
// Used by the `--gpu` encoder and by the tests, both of which this binary can be built
// without: a plain `use` is dead on a `--features vulkan` build of the bin target, which
// is where clippy found it. Gated rather than allow-ed, so the next unused import here is
// still a warning.
#[cfg(any(feature = "rocm", test))]
use rivoli::artifact::quant::VQ_GROUP;
#[cfg(any(feature = "rocm", test))]
use rivoli::math::f32_to_bf16;

const FP8_BLOCK: usize = 128;

/// GPU-accelerated encode: the nearest-codebook argmin (the run-time ceiling, ~VQ_K×
/// the other work) offloaded to `rivoli_vq_encode`. Host still does fp8 dequant,
/// group normalization, the closed-form scale refit, and packing — all O(o·i), a
/// fraction of the argmin. Produces bytes BIT-IDENTICAL to `quant_vq` (same
/// metric/tie-break in the kernel, same host refit here). Ported from `fp82vq`'s
/// validated GPU path; one encoder per projection codebook, shared behind a `Mutex`
/// so host dequant/refit/pack run parallel while the GPU argmins serialize.
#[cfg(feature = "rocm")]
mod gpu {
    use super::*;
    use rivoli::memory::device::DeviceBuf;
    use rivoli::backend::hip::{device_sync, launch_vq_encode};
    use rivoli::artifact::quant::{set_idx, vq_groups};

    fn f32_as_bytes(v: &[f32]) -> &[u8] {
        // SAFETY: f32 is POD; the view is `4·len` read-only bytes (LE host == LE device).
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
    }

    /// Resident codebook + `‖c‖²` on device, plus reusable subvector/index buffers
    /// sized to the largest projection.
    pub struct Encoder {
        codebook: DeviceBuf,
        cbnorm: DeviceBuf,
        sub: DeviceBuf,
        idx: DeviceBuf,
        max_sub: usize,
    }

    // SAFETY: DeviceBuf holds process-global device pointers; every access goes
    // through `encode` behind the caller's Mutex, so device ops never run concurrent.
    unsafe impl Send for Encoder {}

    impl Encoder {
        pub fn new(codebook: &[f32], max_sub: usize) -> Result<Self> {
            let cbnorm = rivoli::artifact::quant::codebook_norms(codebook);
            let mut cb = DeviceBuf::new(codebook.len() * 4)?;
            cb.copy_in_at(0, f32_as_bytes(codebook))?;
            let mut nb = DeviceBuf::new(cbnorm.len() * 4)?;
            nb.copy_in_at(0, f32_as_bytes(&cbnorm))?;
            Ok(Self {
                codebook: cb,
                cbnorm: nb,
                sub: DeviceBuf::new(max_sub * VQ_DIM * 4)?,
                idx: DeviceBuf::new(max_sub * 2)?,
                max_sub,
            })
        }

        pub fn encode(&mut self, sub: &[f32]) -> Result<Vec<u16>> {
            let n = sub.len() / VQ_DIM;
            ensure!(
                n <= self.max_sub,
                "encode batch {n} > max_sub {}",
                self.max_sub
            );
            self.sub.copy_in_at(0, f32_as_bytes(sub))?;
            // SAFETY: buffers sized ≥ n; live until the sync below.
            unsafe {
                launch_vq_encode(
                    self.sub.ptr() as *const f32,
                    self.codebook.ptr() as *const f32,
                    self.cbnorm.ptr() as *const f32,
                    n,
                    self.idx.ptr_mut() as *mut u16,
                )?;
            }
            device_sync()?;
            let mut bytes = Vec::new();
            self.idx.copy_out_prefix(&mut bytes, n * 2)?;
            Ok(bytes
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect())
        }
    }

    /// GPU analog of `quant_vq`: two batched argmin passes (amax-scale, then the
    /// refit scale) around the same host refit. Bit-identical output to `quant_vq`.
    pub fn quant_vq_gpu(
        w: &[f32],
        o_dim: usize,
        i_dim: usize,
        codebook: &[f32],
        enc: &std::sync::Mutex<Encoder>,
    ) -> Result<(Vec<u8>, Vec<u16>)> {
        const SUBS: usize = VQ_GROUP / VQ_DIM;
        let ngroups = i_dim / VQ_GROUP;
        let nsub_row = i_dim / VQ_DIM;
        let rb = vq_row_bytes(i_dim);
        let _ = vq_groups(i_dim); // dim sanity (asserts i_dim % VQ_GROUP == 0)
        let mut sub = vec![0.0f32; o_dim * nsub_row * VQ_DIM];
        let mut scales = vec![0u16; o_dim * ngroups];
        // Pass 1: normalize every group by its amax-derived bf16 scale.
        for o in 0..o_dim {
            let wr = &w[o * i_dim..(o + 1) * i_dim];
            for grp in 0..ngroups {
                let seg = &wr[grp * VQ_GROUP..(grp + 1) * VQ_GROUP];
                let amax = seg.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
                let sb = f32_to_bf16(if amax > 0.0 { amax } else { 1.0 });
                scales[o * ngroups + grp] = sb;
                let inv = 1.0 / bf16_to_f32(sb);
                let base = (o * nsub_row + grp * SUBS) * VQ_DIM;
                for (t, &v) in seg.iter().enumerate() {
                    sub[base + t] = v * inv;
                }
            }
        }
        let idx1 = enc
            .lock()
            .map_err(|_| anyhow::anyhow!("encoder poisoned"))?
            .encode(&sub)?;
        // Refit each group against its pass-1 entries; renormalize only changed groups.
        let mut changed = false;
        for o in 0..o_dim {
            let wr = &w[o * i_dim..(o + 1) * i_dim];
            for grp in 0..ngroups {
                let seg = &wr[grp * VQ_GROUP..(grp + 1) * VQ_GROUP];
                let sbase = o * nsub_row + grp * SUBS;
                // The refit itself is `quant_vq`'s, called rather than restated: the two
                // encoders are asserted bit-identical under `--validate`, and this
                // accumulation is where a divergence would come from.
                let refit = rivoli::artifact::quant::vq_refit(
                    seg,
                    &idx1[sbase..sbase + SUBS],
                    codebook,
                );
                if let Some(refit) = refit
                    && refit != scales[o * ngroups + grp]
                {
                    scales[o * ngroups + grp] = refit;
                    let inv = 1.0 / bf16_to_f32(refit);
                    let base = sbase * VQ_DIM;
                    for (t, &v) in seg.iter().enumerate() {
                        sub[base + t] = v * inv;
                    }
                    changed = true;
                }
            }
        }
        let idx = if changed {
            enc.lock()
                .map_err(|_| anyhow::anyhow!("encoder poisoned"))?
                .encode(&sub)?
        } else {
            idx1
        };
        let mut indices = vec![0u8; o_dim * rb];
        for o in 0..o_dim {
            let ir = &mut indices[o * rb..(o + 1) * rb];
            for t in 0..nsub_row {
                set_idx(ir, t, idx[o * nsub_row + t]);
            }
        }
        Ok((indices, scales))
    }
}

/// Model dims the converter needs (a subset of config.json).
struct Dims {
    hidden: usize,
    moe_inter: usize,
    n_experts: usize,
    n_layers: usize,
    dense_layers: usize,
}

fn dims(cfg: &serde_json::Value) -> Result<Dims> {
    let g = |k: &str| {
        cfg[k]
            .as_u64()
            .with_context(|| format!("config missing {k}"))
            .map(|v| v as usize)
    };
    Ok(Dims {
        hidden: g("hidden_size")?,
        moe_inter: g("moe_intermediate_size")?,
        n_experts: g("n_routed_experts").or_else(|_| g("num_experts"))?,
        n_layers: g("num_hidden_layers")?,
        dense_layers: g("first_k_dense_replace")?,
    })
}

/// Dequantize an fp8 projection `<base>.<proj>.weight` (+ `weight_scale_inv`) to f32.
fn deq(src: &Safetensors, base: &str, proj: &str, o_dim: usize, i_dim: usize) -> Result<Vec<f32>> {
    src.dequant_fp8(&format!("{base}.{proj}"), o_dim, i_dim, FP8_BLOCK)
}

/// Write one projection's (indices, scales) into `dst`: indices then LE bf16 scales.
fn write_proj(dst: &mut [u8], o_dim: usize, i_dim: usize, indices: &[u8], scales: &[u16]) {
    let ib = o_dim * vq_row_bytes(i_dim);
    dst[..ib].copy_from_slice(indices);
    write_le_scales(&mut dst[ib..], scales.iter().map(|s| s.to_le_bytes()));
}

/// Per-projection GPU encoders (`--gpu`): one `Encoder` per codebook, each behind a
/// `Mutex` so the parallel expert encode serializes only the GPU argmin calls. `()`
/// without the `rocm` feature (where `--gpu` is rejected at parse and never built).
#[cfg(feature = "rocm")]
type Enc = [std::sync::Mutex<gpu::Encoder>; 3];
#[cfg(not(feature = "rocm"))]
type Enc = ();

/// Encode one expert (routed or shared) rooted at `base` into `dst`, gate/up/down
/// against `codebooks[0..3]`. With `enc` (the `--gpu` encoders) the nearest-codebook
/// argmin offloads to the GPU (bit-identical to the CPU `quant_vq`).
#[allow(clippy::too_many_arguments)]
fn encode_expert(
    src: &Safetensors,
    base: &str,
    projs: &ExpertProjs,
    codebooks: &[Vec<f32>; 3],
    dst: &mut [u8],
    enc: Option<&Enc>,
    validate: bool,
) -> Result<()> {
    let mut off = 0;
    for (p, &(proj, (o_dim, i_dim))) in projs.iter().enumerate() {
        let w = deq(src, base, proj, o_dim, i_dim)?;
        let (indices, scales) = match enc {
            #[cfg(feature = "rocm")]
            Some(e) => {
                let g = gpu::quant_vq_gpu(&w, o_dim, i_dim, &codebooks[p], &e[p])?;
                // `--validate`: the GPU argmin must reproduce the CPU quant_vq
                // bit-for-bit (same metric, tie-break, and host refit) — else the two
                // paths would silently disagree on the shipped bytes.
                if validate {
                    let c = quant_vq(&w, o_dim, i_dim, &codebooks[p]);
                    ensure!(g == c, "GPU/CPU encode mismatch at {base} {proj}");
                }
                g
            }
            _ => quant_vq(&w, o_dim, i_dim, &codebooks[p]),
        };
        let pb = vq_proj_bytes(o_dim, i_dim);
        write_proj(&mut dst[off..off + pb], o_dim, i_dim, &indices, &scales);
        off += pb;
    }
    let _ = validate; // consumed above only under `rocm`; silence the unused warning
    Ok(())
}

/// Requantize a bf16 `[o_dim, i_dim]` matrix to per-row int8 → (packed i8, f32 scale).
fn requant_int8(bytes: &[u8], o_dim: usize, i_dim: usize) -> (Vec<u8>, Vec<u8>) {
    let mut packed = vec![0u8; o_dim * i_dim];
    let mut scales = Vec::with_capacity(o_dim * 4);
    for o in 0..o_dim {
        let row = &bytes[o * i_dim * 2..(o + 1) * i_dim * 2];
        let vals: Vec<f32> = row
            .chunks_exact(2)
            .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        let amax = vals.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let s = if amax > 0.0 { amax / 127.0 } else { 1.0 };
        scales.extend(s.to_le_bytes());
        for (i, &v) in vals.iter().enumerate() {
            packed[o * i_dim + i] = ((v / s).round().clamp(-127.0, 127.0) as i8) as u8;
        }
    }
    (packed, scales)
}

/// Add a bf16 `[o,i]` tensor requantized to int8 (embed, lm_head).
fn add_int8(src: &Safetensors, w: &mut SafeWriter, name: &str) -> Result<()> {
    let (bytes, shape) = src.typed(name, Dtype::Bf16)?;
    let (o, i) = (shape[0], shape[1]);
    let (packed, scales) = requant_int8(bytes, o, i);
    w.add(name, Dtype::I8, vec![o, i], packed);
    w.add(format!("{name}.scale"), Dtype::F32, vec![o], scales);
    Ok(())
}

// clap derives the parse and the help text from this struct — the same switch src/main.rs
// made, for the same reason: the hand-rolled `std::env::args()` loop this replaced kept a
// usage string maintained separately from the flags it described, and a second source of
// truth for a flag list only ever drifts apart from the first. This one already had:
// its `usage:` line named neither `--gpu` nor `--validate`, the two flags that decide
// whether a full convert takes an hour or a week.
//
// NOTE: doc comments on the FIELDS below are USER-FACING — clap renders them as `--help`.
// Rationale for the code goes in `//` comments like this one, which clap ignores.
#[derive(Parser)]
#[command(
    name = "convert",
    about = "GLM-5.2 fp8 checkpoint → the rivoli int3-vq artifact (.vq3 experts, resident set, manifest)"
)]
struct Args {
    /// The fp8 GLM-5.2 checkpoint directory: config.json, the `*.safetensors` shards,
    /// and the tokenizer files that get copied into the artifact.
    fp8_dir: String,

    /// Artifact directory to write — manifest.json, codebooks.f32, resident.safetensors,
    /// one `L{ll}.vq3` per MoE layer, and the copied tokenizer. Created if absent, and an
    /// existing codebooks.f32 or `L{ll}.vq3` is REUSED rather than re-encoded, so a
    /// killed run resumes by re-running the same command line.
    out_dir: String,

    /// Convert only the first N MoE layers (the dense prefix is always carried) — an
    /// artifact/loader smoke test in minutes instead of hours. The manifest records the
    /// reduced layer count, and the MTP head is dropped: it has no hidden state to consume
    /// in a truncated model.
    #[arg(long, value_name = "N")]
    layers: Option<usize>,

    /// How many layers to draw codebook training subvectors from, one expert each, spread
    /// evenly across the MoE range. Ignored when codebooks.f32 already exists.
    #[arg(long, value_name = "N", default_value_t = 48)]
    sample_experts: usize,

    /// k-means iterations per projection codebook. Ignored when codebooks.f32 exists.
    #[arg(long, value_name = "N", default_value_t = 40)]
    kmeans_iters: usize,

    /// Offload the nearest-codebook argmin to the GPU (needs `--features rocm`) — it is
    /// ~VQ_K× the encode work, so this is ~an hour for the full model against many on
    /// CPU. The bytes it writes are bit-identical to the CPU encoder's.
    #[arg(long)]
    gpu: bool,

    /// Cross-check every GPU-encoded projection against the CPU `quant_vq` (bytes must
    /// match exactly). Roughly doubles encode time. Only meaningful with `--gpu`, which
    /// it therefore requires.
    #[arg(long, requires = "gpu")]
    validate: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let cfg: serde_json::Value =
        serde_json::from_slice(&std::fs::read(format!("{}/config.json", args.fp8_dir))?)?;
    let d = dims(&cfg)?;
    let last = args
        .layers
        .map_or(d.n_layers, |n| (d.dense_layers + n).min(d.n_layers));
    std::fs::create_dir_all(&args.out_dir)?;
    let src = Safetensors::open_dir(&args.fp8_dir)?;
    eprintln!(
        "convert: hidden={} moe_inter={} experts={} layers {}..{} (dense<{})",
        d.hidden, d.moe_inter, d.n_experts, d.dense_layers, last, d.dense_layers
    );

    // 1. Learn the 3 per-projection codebooks (reuse an existing codebooks.f32).
    let cb_path = format!("{}/codebooks.f32", args.out_dir);
    let codebooks: [Vec<f32>; 3] = if let Ok(b) = std::fs::read(&cb_path) {
        let raw = read_f32(&b);
        ensure!(raw.len() == 3 * VQ_K * VQ_DIM, "bad codebooks.f32");
        let n = VQ_K * VQ_DIM;
        eprintln!("convert: reusing {cb_path}");
        [
            raw[..n].to_vec(),
            raw[n..2 * n].to_vec(),
            raw[2 * n..].to_vec(),
        ]
    } else {
        const TARGET: usize = 1 << 20;
        let per_expert = d.moe_inter * d.hidden / VQ_DIM; // per projection
        let mut cbs: [Vec<f32>; 3] = [vec![], vec![], vec![]];
        for (p, &(proj, (o_dim, i_dim))) in expert_projs(d.hidden, d.moe_inter).iter().enumerate() {
            let mut sample = Vec::new();
            let layers: Vec<usize> = (d.dense_layers..last)
                .step_by(((last - d.dense_layers) / args.sample_experts).max(1))
                .collect();
            let stride = (layers.len() * per_expert / TARGET).max(1);
            for &l in &layers {
                let e = l % d.n_experts;
                let w = deq(
                    &src,
                    &format!("model.layers.{l}.mlp.experts.{e}"),
                    proj,
                    o_dim,
                    i_dim,
                )?;
                sample_subvectors(&w, i_dim, stride, &mut sample);
            }
            eprintln!(
                "convert: learning {} codebook from {} subvectors…",
                proj,
                sample.len() / VQ_DIM
            );
            cbs[p] = learn_codebook(&sample, args.kmeans_iters);
        }
        let mut bytes = Vec::with_capacity(3 * VQ_K * VQ_DIM * 4);
        for cb in &cbs {
            for &v in cb {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        std::fs::write(&cb_path, &bytes)?;
        cbs
    };

    // 2. Encode experts → per-layer .vq3 (header + routed + shared block). With
    // `--gpu`, build one resident encoder per projection codebook (reused across all
    // layers); the parallel encode then offloads only the argmin.
    #[cfg(feature = "rocm")]
    let encoders: Option<Enc> = if args.gpu {
        let max_sub = d.hidden * d.moe_inter / VQ_DIM; // largest projection subvectors
        eprintln!("convert: GPU encode enabled (argmin offloaded to vq_encode)");
        Some([
            std::sync::Mutex::new(gpu::Encoder::new(&codebooks[0], max_sub)?),
            std::sync::Mutex::new(gpu::Encoder::new(&codebooks[1], max_sub)?),
            std::sync::Mutex::new(gpu::Encoder::new(&codebooks[2], max_sub)?),
        ])
    } else {
        None
    };
    #[cfg(not(feature = "rocm"))]
    let encoders: Option<Enc> = {
        ensure!(!args.gpu, "--gpu requires building with --features rocm");
        None
    };

    // The MTP head (`num_nextn_predict_layers`) is checkpoint layer `n_layers`: a full
    // MoE layer plus enorm/hnorm/eh_proj/shared_head.norm. Carried only on a whole-model
    // convert — a `--layers`-limited artifact has no hidden state for it to consume.
    let mtp = args.layers.is_none() && src.has(&format!("model.layers.{}.eh_proj.weight", d.n_layers));
    eprintln!("convert: mtp head {}", if mtp { "carried" } else { "absent" });
    let moe_layers: Vec<usize> = (d.dense_layers..last).chain(mtp.then_some(d.n_layers)).collect();

    let stride = rivoli::artifact::quant::vq_expert_stride(d.hidden, d.moe_inter);
    let ebytes = vq_expert_bytes(d.hidden, d.moe_inter);
    // Hoisted out of the per-expert worker loop: the name/shape pairing is the same for
    // every expert of every layer.
    let projs = expert_projs(d.hidden, d.moe_inter);
    for &l in &moe_layers {
        let path = format!("{}/L{l:02}.vq3", args.out_dir);
        if std::fs::metadata(&path).is_ok() {
            continue;
        }
        let mut buf = vec![0u8; VQ_ALIGN + (d.n_experts + 1) * stride];
        let hdr = Vq3Header::new(l, d.n_experts, d.hidden, d.moe_inter).to_bytes();
        buf[..hdr.len()].copy_from_slice(&hdr);
        // Encode all n_experts+1 blocks (routed 0..n, shared = n) in parallel over
        // disjoint block slices — quant_vq is pure and Safetensors is Sync.
        let (src, cb, projs) = (&src, &codebooks, &projs);
        let enc = encoders.as_ref();
        let validate = args.validate;
        let threads = std::thread::available_parallelism().map_or(4, |t| t.get());
        let per = (d.n_experts + 1).div_ceil(threads);
        std::thread::scope(|s| -> Result<()> {
            let mut rest = &mut buf[VQ_ALIGN..];
            let mut e0 = 0;
            let mut handles = Vec::new();
            while e0 <= d.n_experts {
                let take = per.min(d.n_experts + 1 - e0);
                let (mine, tail) = rest.split_at_mut(take * stride);
                rest = tail;
                let base_e = e0;
                e0 += take;
                handles.push(s.spawn(move || -> Result<()> {
                    for (j, slot) in mine.chunks_exact_mut(stride).enumerate() {
                        let e = base_e + j;
                        let base = expert_base(l, e, d.n_experts);
                        encode_expert(src, &base, projs, cb, &mut slot[..ebytes], enc, validate)?;
                    }
                    Ok(())
                }));
            }
            for h in handles {
                h.join()
                    .map_err(|_| anyhow::anyhow!("encode worker panicked"))??;
            }
            Ok(())
        })
        .with_context(|| format!("encode layer {l}"))?;
        std::fs::write(&path, &buf)?;
        eprintln!("convert: wrote {path}");
    }

    // 3. Resident set → resident.safetensors.
    let mut w = SafeWriter::new();
    add_int8(&src, &mut w, "model.embed_tokens.weight")?;
    add_int8(&src, &mut w, "lm_head.weight")?;
    w.add_widened(&src, "model.norm.weight")?;
    for l in (0..last).chain(mtp.then_some(d.n_layers)) {
        let lb = format!("model.layers.{l}");
        w.add_widened(&src, &format!("{lb}.input_layernorm.weight"))?;
        w.add_widened(&src, &format!("{lb}.post_attention_layernorm.weight"))?;
        w.add_widened(&src, &format!("{lb}.self_attn.q_a_layernorm.weight"))?;
        w.add_widened(&src, &format!("{lb}.self_attn.kv_a_layernorm.weight"))?;
        for p in [
            "q_a_proj",
            "q_b_proj",
            "kv_a_proj_with_mqa",
            "kv_b_proj",
            "o_proj",
        ] {
            w.copy_fp8(&src, &format!("{lb}.self_attn.{p}"))?;
        }
        if l < d.dense_layers {
            for p in PROJ {
                w.copy_fp8(&src, &format!("{lb}.mlp.{p}"))?;
            }
        } else {
            w.add_widened(&src, &format!("{lb}.mlp.gate.weight"))?;
            let (bias, bsh) = src.typed(
                &format!("{lb}.mlp.gate.e_score_correction_bias"),
                Dtype::F32,
            )?;
            w.add(
                format!("{lb}.mlp.gate.e_score_correction_bias"),
                Dtype::F32,
                bsh.to_vec(),
                bias.to_vec(),
            );
        }
        // The MTP-only tensors. `eh_proj` is [hidden, 2·hidden] bf16 in the source with
        // no block scales, so it cannot ride `copy_fp8`; widened it is 302 MB against a
        // 15 GB resident set and reuses gemv_f32.
        // ponytail: f32 eh_proj, int8 it if the resident budget ever bites.
        if l == d.n_layers {
            for t in ["enorm.weight", "hnorm.weight", "eh_proj.weight", "shared_head.norm.weight"] {
                w.add_widened(&src, &format!("{lb}.{t}"))?;
            }
        }
    }
    let rpath = format!("{}/resident.safetensors", args.out_dir);
    w.write(&rpath)?;
    eprintln!("convert: wrote {rpath}");

    // 4. manifest.json = source config + a `format` section. A `--layers`-limited
    // convert only writes layers `0..last`, so report the reduced count — the loader
    // iterates `0..num_hidden_layers` and would otherwise demand absent layer files.
    let mut manifest = cfg.clone();
    if args.layers.is_some() {
        manifest["num_hidden_layers"] = serde_json::json!(last);
    }
    manifest["num_nextn_predict_layers"] = serde_json::json!(u8::from(mtp));
    manifest["format"] = serde_json::to_value(FormatMeta::current(FP8_BLOCK))?;
    std::fs::write(
        format!("{}/manifest.json", args.out_dir),
        serde_json::to_vec_pretty(&manifest)?,
    )?;

    // 5. Copy the tokenizer files so the artifact is self-contained (the runtime
    // loads tokenizer.json + generation_config.json from the model dir).
    for name in ["tokenizer.json", "generation_config.json"] {
        let src = format!("{}/{name}", args.fp8_dir);
        let dst = format!("{}/{name}", args.out_dir);
        match std::fs::copy(&src, &dst) {
            Ok(_) => eprintln!("convert: copied {name}"),
            Err(e) => eprintln!("convert: WARNING: {name} not copied ({e})"),
        }
    }
    eprintln!("convert: done — {} → {}", args.fp8_dir, args.out_dir);
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn kmeans_recovers_separated_clusters() {
        let mut sample = Vec::new();
        for i in 0..(VQ_K * 4) {
            let c = if i % 2 == 0 { -10.0 } else { 10.0 };
            for _ in 0..VQ_DIM {
                sample.push(c + ((i as f32 * 7.0).sin()) * 0.01);
            }
        }
        let cb = learn_codebook(&sample, 5);
        for j in 0..VQ_K {
            let m = cb[j * VQ_DIM];
            assert!(
                (m + 10.0).abs() < 0.5 || (m - 10.0).abs() < 0.5,
                "centroid {j}={m}"
            );
        }
    }

    #[test]
    fn requant_int8_roundtrips_within_step() {
        // bf16 [2,4] → int8; dequant within one quant step of the original.
        let vals = [1.0f32, -2.0, 0.5, -0.25, 3.0, -3.0, 0.1, 2.0];
        let bytes: Vec<u8> = vals
            .iter()
            .flat_map(|&v| f32_to_bf16(v).to_le_bytes())
            .collect();
        let (packed, scales) = requant_int8(&bytes, 2, 4);
        for o in 0..2 {
            let s = f32::from_le_bytes(scales[o * 4..o * 4 + 4].try_into().unwrap());
            for i in 0..4 {
                let deq = (packed[o * 4 + i] as i8) as f32 * s;
                assert!((deq - bf16_to_f32(f32_to_bf16(vals[o * 4 + i]))).abs() <= s + 1e-3);
            }
        }
    }

    #[test]
    fn expert_block_slices_via_loader() {
        // Encode a synthetic expert against tiny codebooks, then slice it with the
        // loader's vq_expert and check GEMV reproduces — the write↔read contract.
        use rivoli::artifact::quant::{matvec_vq, vq_expert};
        let (hidden, moe_inter) = (VQ_GROUP, VQ_GROUP);
        let mut cbs: [Vec<f32>; 3] = std::array::from_fn(|p| {
            let mut cb = vec![1e30f32; VQ_K * VQ_DIM];
            cb[0..VQ_DIM].copy_from_slice(&[1.0, -1.0, 0.5 + p as f32, -0.5]);
            cb[VQ_DIM..2 * VQ_DIM].copy_from_slice(&[0.25, 0.5, 0.75, 1.0]);
            cb
        });
        // build the block directly (no fp8 source): quant each projection, write in.
        let ebytes = vq_expert_bytes(hidden, moe_inter);
        let mut block = vec![0u8; ebytes];
        let mut off = 0;
        let mut originals = Vec::new();
        for (p, &(o_dim, i_dim)) in vq_expert_layout(hidden, moe_inter).iter().enumerate() {
            let mut wv = vec![0.0f32; o_dim * i_dim];
            for (n, x) in wv.iter_mut().enumerate() {
                *x = cbs[p][(n % 2) * VQ_DIM + (n % VQ_DIM)] * 0.5;
            }
            let (indices, scales) = quant_vq(&wv, o_dim, i_dim, &cbs[p]);
            write_proj(
                &mut block[off..off + vq_proj_bytes(o_dim, i_dim)],
                o_dim,
                i_dim,
                &indices,
                &scales,
            );
            off += vq_proj_bytes(o_dim, i_dim);
            originals.push((indices, scales, o_dim, i_dim));
        }
        let projs = vq_expert(&block, 0, hidden, moe_inter);
        for (k, (proj, (indices, scales, o_dim, i_dim))) in projs.iter().zip(&originals).enumerate()
        {
            let x: Vec<f32> = (0..*i_dim).map(|v| (v + 1) as f32).collect();
            let mut yl = vec![0.0f32; *o_dim];
            let mut yr = vec![0.0f32; *o_dim];
            let ps: Vec<u16> =
                proj.scales.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
            matvec_vq(&mut yl, &x, proj.indices, &ps, &cbs[k], proj.o_dim, proj.i_dim);
            matvec_vq(&mut yr, &x, indices, scales, &cbs[k], *o_dim, *i_dim);
            assert_eq!(yl, yr);
        }
        let _ = &mut cbs;
    }
}
