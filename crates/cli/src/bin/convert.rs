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
use rivoli_artifact::format::{
    Dtype, ExpertHeader, FormatMeta, LAYER_WINDOW, LayerDims, SafeWriter, Safetensors, VQ3_MAGIC,
    finish_artifact, load_codebooks, write_expert_layer,
};
use rivoli_artifact::quant::{
    ExpertProjs, FP8_BLOCK, PROJ, VQ_DIM, expert_base, expert_projs, learn_codebook, quant_vq,
    sample_subvectors, vq_expert_bytes, vq_proj_bytes, vq_row_bytes, write_le_scales,
};
use rivoli_core::num::bf16_to_f32;
// Only the tests still take the layout on its own; the encoders go through `expert_projs`,
// which carries the projection NAMES alongside it. `VqW` is the same case — only the
// round-trip test GEMVs the bytes back — and so is `VQ_K`: every shipping path takes the
// entry count from the codebooks it was handed. Gated rather than allow-ed, for the same
// reason as `VQ_GROUP` below.
#[cfg(test)]
use rivoli_artifact::quant::{VQ_K, VqW, vq_expert_layout};
// Used by the `--gpu` encoder and by the tests, both of which this binary can be built
// without: a plain `use` is dead on a `--features vulkan` build of the bin target, which
// is where clippy found it. Gated rather than allow-ed, so the next unused import here is
// still a warning.
#[cfg(any(feature = "rocm", test))]
use rivoli_artifact::quant::VQ_GROUP;
#[cfg(any(feature = "rocm", test))]
use rivoli_core::num::f32_to_bf16;

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
    use rivoli_artifact::quant::{set_idx, vq_groups};
    use rivoli_backend::hip::{device_sync, launch_vq_encode};
    use rivoli_engine::device::DeviceBuf;

    /// Subvectors per group — the span one bf16 scale covers, and therefore the slice of
    /// pass-1 indices the refit is fitted against.
    const SUBS: usize = VQ_GROUP / VQ_DIM;

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
            let cbnorm = rivoli_artifact::quant::codebook_norms(codebook);
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

    /// One projection's geometry, derived once. Both passes and the packer index the same
    /// three arrays off it, so a recomputation is a chance for them to disagree.
    #[derive(Clone, Copy)]
    struct Shape {
        o_dim: usize,
        i_dim: usize,
        ngroups: usize,
        nsub_row: usize,
    }

    impl Shape {
        /// Every group of `w` as `(weights, staging subvector base, scale index)`, in
        /// row-major order. Stated once because both passes walk it and a refit is only
        /// sound if it renormalizes the exact span pass 1 normalized.
        ///
        /// By value, not by reference: the caller reads it out of the `Staging` it is about
        /// to write, so a borrow here would collide with that write.
        fn groups(self, w: &[f32]) -> impl Iterator<Item = (&[f32], usize, usize)> {
            let Shape {
                o_dim,
                i_dim,
                ngroups,
                nsub_row,
            } = self;
            (0..o_dim).flat_map(move |o| {
                (0..ngroups).map(move |grp| {
                    (
                        &w[o * i_dim + grp * VQ_GROUP..][..VQ_GROUP],
                        o * nsub_row + grp * SUBS,
                        o * ngroups + grp,
                    )
                })
            })
        }
    }

    /// What the host hands the GPU and what it keeps: `sub` is every group's weights
    /// divided by its scale (the argmin input), `scales` the bf16 scale each was divided
    /// by. One value because they are only meaningful together — a scale that moves
    /// invalidates exactly its own subvectors.
    struct Staging {
        sh: Shape,
        sub: Vec<f32>,
        scales: Vec<u16>,
    }

    impl Staging {
        /// Derives the geometry as well as the arrays: `Shape` has no other construction
        /// site, and one constructor is one place for the two to disagree about the dims.
        fn new(shape: [usize; 2]) -> Self {
            let [o_dim, i_dim] = shape;
            let _ = vq_groups(i_dim); // dim sanity (asserts i_dim % VQ_GROUP == 0)
            let sh = Shape {
                o_dim,
                i_dim,
                ngroups: i_dim / VQ_GROUP,
                nsub_row: i_dim / VQ_DIM,
            };
            Staging {
                sh,
                sub: vec![0.0f32; sh.o_dim * sh.nsub_row * VQ_DIM],
                scales: vec![0u16; sh.o_dim * sh.ngroups],
            }
        }

        /// Divide one group by `scale` into its subvector span — the only write into `sub`,
        /// so pass 1 and the refit cannot normalize differently.
        fn renorm(&mut self, seg: &[f32], sbase: usize, scale: u16) {
            let inv = 1.0 / bf16_to_f32(scale);
            let base = sbase * VQ_DIM;
            for (t, &v) in seg.iter().enumerate() {
                self.sub[base + t] = v * inv;
            }
        }

        /// Pass 1: normalize every group by its amax-derived bf16 scale.
        fn normalize(&mut self, w: &[f32]) {
            for (seg, sbase, si) in self.sh.groups(w) {
                let amax = seg.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
                let sb = f32_to_bf16(if amax > 0.0 { amax } else { 1.0 });
                self.scales[si] = sb;
                self.renorm(seg, sbase, sb);
            }
        }

        /// Refit each group against its pass-1 entries; renormalize only changed groups.
        /// Reports whether any changed — if none did, pass 1's indices are already final
        /// and the second GPU argmin is skipped entirely.
        fn refit(&mut self, w: &[f32], codebook: &[f32], idx1: &[u16]) -> bool {
            let mut changed = false;
            for (seg, sbase, si) in self.sh.groups(w) {
                // The refit itself is `quant_vq`'s, called rather than restated: the two
                // encoders are asserted bit-identical under `--validate`, and this
                // accumulation is where a divergence would come from.
                let refit =
                    rivoli_artifact::quant::vq_refit(seg, &idx1[sbase..sbase + SUBS], codebook);
                if let Some(refit) = refit
                    && refit != self.scales[si]
                {
                    self.scales[si] = refit;
                    self.renorm(seg, sbase, refit);
                    changed = true;
                }
            }
            changed
        }

        /// Flat per-subvector indices → the on-disk row layout (`set_idx` owns the 12-bit
        /// packing, so this only walks it), paired with the scales they were fitted under.
        fn pack(self, idx: &[u16]) -> (Vec<u8>, Vec<u16>) {
            let (rb, nsub_row) = (vq_row_bytes(self.sh.i_dim), self.sh.nsub_row);
            let mut indices = vec![0u8; self.sh.o_dim * rb];
            for (o, ir) in indices.chunks_exact_mut(rb).enumerate() {
                for t in 0..nsub_row {
                    set_idx(ir, t, idx[o * nsub_row + t]);
                }
            }
            (indices, self.scales)
        }
    }

    /// One locked argmin batch. Both call sites go through it so the poison mapping — a
    /// worker that died mid-launch, from which there is no recovery here — is written once.
    fn argmin(enc: &std::sync::Mutex<Encoder>, sub: &[f32]) -> Result<Vec<u16>> {
        enc.lock()
            .map_err(|_| anyhow::anyhow!("encoder poisoned"))?
            .encode(sub)
    }

    /// GPU analog of `quant_vq`: two batched argmin passes (amax-scale, then the
    /// refit scale) around the same host refit. Bit-identical output to `quant_vq`.
    pub fn quant_vq_gpu(
        w: &[f32],
        shape: [usize; 2],
        codebook: &[f32],
        enc: &std::sync::Mutex<Encoder>,
    ) -> Result<(Vec<u8>, Vec<u16>)> {
        let mut st = Staging::new(shape);
        st.normalize(w);
        let idx1 = argmin(enc, &st.sub)?;
        let idx = if st.refit(w, codebook, &idx1) {
            argmin(enc, &st.sub)?
        } else {
            idx1
        };
        Ok(st.pack(&idx))
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
fn deq(src: &Safetensors, base: &str, proj: &str, shape: [usize; 2]) -> Result<Vec<f32>> {
    src.dequant_fp8(&format!("{base}.{proj}"), shape[0], shape[1], FP8_BLOCK)
}

/// Write one projection's (indices, scales) into `dst`: indices then LE bf16 scales.
fn write_proj(dst: &mut [u8], shape: [usize; 2], indices: &[u8], scales: &[u16]) {
    let ib = shape[0] * vq_row_bytes(shape[1]);
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

/// Everything an expert encode holds fixed for a whole run: the opened checkpoint, the 3
/// codebooks, the projection name/shape table, and the `--gpu`/`--validate` decisions. One
/// value rather than a parameter list, so the per-expert call carries only what varies —
/// which expert, and which slot of the layer buffer it lands in.
struct Encode<'a> {
    src: &'a Safetensors,
    codebooks: &'a [Vec<f32>; 3],
    projs: &'a ExpertProjs,
    enc: Option<&'a Enc>,
    validate: bool,
}

impl Encode<'_> {
    /// Encode one expert (routed or shared) rooted at `base` into `dst`: each projection
    /// dequantized from the fp8 source, VQ-encoded, and written indices-then-scales into
    /// its own `vq_proj_bytes` span, gate/up/down back to back.
    fn expert(&self, base: &str, dst: &mut [u8]) -> Result<()> {
        let mut off = 0;
        for (p, &(proj, (o_dim, i_dim))) in self.projs.iter().enumerate() {
            let pb = vq_proj_bytes(o_dim, i_dim);
            let w = deq(self.src, base, proj, [o_dim, i_dim])?;
            let (indices, scales) = self.quantize(&w, p, base)?;
            write_proj(&mut dst[off..off + pb], [o_dim, i_dim], &indices, &scales);
            off += pb;
        }
        Ok(())
    }

    /// `quant_vq` against `codebooks[p]` — offloaded to the GPU when `--gpu` built the
    /// encoders (bit-identical to the CPU path). `base` names the tensor in the
    /// `--validate` mismatch report.
    fn quantize(&self, w: &[f32], p: usize, base: &str) -> Result<(Vec<u8>, Vec<u16>)> {
        let (proj, (o_dim, i_dim)) = self.projs[p];
        let cb = &self.codebooks[p];
        // Read only under `rocm`, where `--gpu` can be accepted; without it the CPU arm
        // below is the only one and the compiler would call all three dead.
        let _ = (self.validate, proj, base);
        match self.enc {
            #[cfg(feature = "rocm")]
            Some(e) => {
                let g = gpu::quant_vq_gpu(w, [o_dim, i_dim], cb, &e[p])?;
                // `--validate`: the GPU argmin must reproduce the CPU quant_vq
                // bit-for-bit (same metric, tie-break, and host refit) — else the two
                // paths would silently disagree on the shipped bytes.
                if self.validate {
                    let c = quant_vq(w, o_dim, i_dim, cb);
                    ensure!(g == c, "GPU/CPU encode mismatch at {base} {proj}");
                }
                Ok(g)
            }
            _ => Ok(quant_vq(w, o_dim, i_dim, cb)),
        }
    }
}

/// Requantize a bf16 `[o_dim, i_dim]` matrix to per-row int8 → (packed i8, f32 scale). An
/// all-zero row takes step 1.0 rather than 0.0: both dequantize to zeros, but a zero step
/// would make the encode `0/0`.
fn requant_int8(bytes: &[u8], o_dim: usize, i_dim: usize) -> (Vec<u8>, Vec<u8>) {
    let mut packed = vec![0u8; o_dim * i_dim];
    let mut scales = Vec::with_capacity(o_dim * 4);
    let rows = bytes
        .chunks_exact(i_dim * 2)
        .zip(packed.chunks_exact_mut(i_dim));
    for (row, out) in rows {
        let w = row
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]));
        let vals: Vec<f32> = w.map(bf16_to_f32).collect();
        let amax = vals.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let s = if amax > 0.0 { amax / 127.0 } else { 1.0 };
        for (o, &v) in out.iter_mut().zip(&vals) {
            *o = ((v / s).round().clamp(-127.0, 127.0) as i8) as u8;
        }
        scales.extend(s.to_le_bytes());
    }
    (packed, scales)
}

/// What `load_codebooks` reads: the 3 codebooks concatenated as little-endian f32. The
/// split side is NOT restated here — `format::load_codebooks` owns it, and the copy this
/// converter used to keep was a second statement of the file's layout (jscpd, 2026-08-15).
fn codebook_bytes(cbs: &[Vec<f32>; 3]) -> Vec<u8> {
    cbs.iter().flatten().flat_map(|v| v.to_le_bytes()).collect()
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

/// The four values every stage below reads: the parsed flags, the model dims, the opened
/// checkpoint, and the exclusive layer bound `--layers` produced. They travel together
/// through all five stages, so they travel as one value.
struct Job<'a> {
    args: &'a Args,
    d: &'a Dims,
    src: &'a Safetensors,
    last: usize,
}

impl<'a> Job<'a> {
    /// The MTP head (`num_nextn_predict_layers`) is checkpoint layer `n_layers`: a full
    /// MoE layer plus enorm/hnorm/eh_proj/shared_head.norm. Carried only on a whole-model
    /// convert — a `--layers`-limited artifact has no hidden state for it to consume.
    fn mtp(&self) -> bool {
        self.args.layers.is_none()
            && self
                .src
                .has(&format!("model.layers.{}.eh_proj.weight", self.d.n_layers))
    }

    /// Stage 1 — the 3 per-projection codebooks, reusing an existing codebooks.f32. That
    /// reuse is what makes re-running the same command line a resume, not a restart.
    ///
    /// Presence decides, and the loader then decides validity: a file that is there but
    /// wrong is an error rather than a silent re-learn that would overwrite it.
    fn codebooks(&self) -> Result<[Vec<f32>; 3]> {
        let path = format!("{}/codebooks.f32", self.args.out_dir);
        if std::fs::metadata(&path).is_ok() {
            eprintln!("convert: reusing {path}");
            return load_codebooks(&self.args.out_dir);
        }
        let cbs = self.learn_codebooks()?;
        std::fs::write(&path, codebook_bytes(&cbs))?;
        Ok(cbs)
    }

    /// One codebook fitted per projection, each from its own sample.
    fn learn_codebooks(&self) -> Result<[Vec<f32>; 3]> {
        let mut cbs: [Vec<f32>; 3] = [vec![], vec![], vec![]];
        let projs = expert_projs(self.d.hidden, self.d.moe_inter);
        for (p, &(proj, (o_dim, i_dim))) in projs.iter().enumerate() {
            let sample = self.sample_proj(proj, [o_dim, i_dim])?;
            eprintln!(
                "convert: learning {} codebook from {} subvectors…",
                proj,
                sample.len() / VQ_DIM
            );
            cbs[p] = learn_codebook(&sample, self.args.kmeans_iters);
        }
        Ok(cbs)
    }

    /// Training subvectors for one projection: one expert from each of `--sample-experts`
    /// layers spread evenly over the MoE range, then strided down to ~TARGET so the fit
    /// costs the same whatever the layer count.
    fn sample_proj(&self, proj: &str, shape: [usize; 2]) -> Result<Vec<f32>> {
        const TARGET: usize = 1 << 20;
        let per_expert = self.d.moe_inter * self.d.hidden / VQ_DIM; // per projection
        let span = self.last - self.d.dense_layers;
        let layers: Vec<usize> = (self.d.dense_layers..self.last)
            .step_by((span / self.args.sample_experts).max(1))
            .collect();
        let stride = (layers.len() * per_expert / TARGET).max(1);
        let mut sample = Vec::new();
        for &l in &layers {
            let e = l % self.d.n_experts;
            let base = format!("model.layers.{l}.mlp.experts.{e}");
            let w = deq(self.src, &base, proj, shape)?;
            sample_subvectors(&w, shape[1], stride, &mut sample);
        }
        Ok(sample)
    }

    /// Stage 2a — one resident GPU encoder per projection codebook (`--gpu`), reused
    /// across every layer; the parallel encode then offloads only the argmin.
    #[cfg(feature = "rocm")]
    fn encoders(&self, codebooks: &[Vec<f32>; 3]) -> Result<Option<Enc>> {
        if !self.args.gpu {
            return Ok(None);
        }
        let max_sub = self.d.hidden * self.d.moe_inter / VQ_DIM; // largest projection subvectors
        eprintln!("convert: GPU encode enabled (argmin offloaded to vq_encode)");
        Ok(Some([
            std::sync::Mutex::new(gpu::Encoder::new(&codebooks[0], max_sub)?),
            std::sync::Mutex::new(gpu::Encoder::new(&codebooks[1], max_sub)?),
            std::sync::Mutex::new(gpu::Encoder::new(&codebooks[2], max_sub)?),
        ]))
    }

    /// Without the `rocm` feature there is no encoder to build, so `--gpu` is a refusal
    /// rather than a silent CPU fallback taking a week.
    #[cfg(not(feature = "rocm"))]
    fn encoders(&self, codebooks: &[Vec<f32>; 3]) -> Result<Option<Enc>> {
        let _ = codebooks;
        ensure!(
            !self.args.gpu,
            "--gpu requires building with --features rocm"
        );
        Ok(None)
    }

    /// Stage 2b — experts encoded into per-layer .vq3 (header + routed + shared block),
    /// skipping any layer a killed run already finished.
    fn encode_layers(&self, codebooks: &[Vec<f32>; 3], enc: Option<&Enc>, mtp: bool) -> Result<()> {
        // Hoisted out of the per-expert worker loop: the name/shape pairing is the same for
        // every expert of every layer.
        let projs = expert_projs(self.d.hidden, self.d.moe_inter);
        let cx = Encode {
            src: self.src,
            codebooks,
            projs: &projs,
            enc,
            validate: self.args.validate,
        };
        for l in (self.d.dense_layers..self.last).chain(mtp.then_some(self.d.n_layers)) {
            let path = format!("{}/L{l:02}.vq3", self.args.out_dir);
            if std::fs::metadata(&path).is_ok() {
                continue;
            }
            self.write_layer(&path, l, &cx)?;
            eprintln!("convert: wrote {path}");
        }
        Ok(())
    }

    /// One layer's `n_experts + 1` blocks (routed `0..n`, shared at `n`), encoded in
    /// parallel over disjoint block slices — `quant_vq` is pure and `Safetensors` is Sync.
    /// Atomic and bounded-memory, and the `continue` in the caller is why the first
    /// matters — see `write_expert_layer`.
    fn write_layer(&self, path: &str, l: usize, cx: &Encode<'_>) -> Result<()> {
        let d = self.d;
        let stride = rivoli_artifact::quant::vq_expert_stride(d.hidden, d.moe_inter);
        let header = ExpertHeader::new(
            VQ3_MAGIC,
            LayerDims {
                layer: l,
                n_experts: d.n_experts,
                expert_in: d.hidden,
                moe_inter: d.moe_inter,
                stride: stride,
            },
        )
        .to_bytes();
        write_expert_layer(
            path,
            &header,
            stride,
            vq_expert_bytes(d.hidden, d.moe_inter),
            d.n_experts + 1,
            LAYER_WINDOW,
            |e, slot| cx.expert(&expert_base(l, e, d.n_experts), slot),
        )
        .with_context(|| format!("write layer {l} (encode or I/O)"))?;
        Ok(())
    }

    /// Stage 3 — the resident set, into resident.safetensors.
    fn write_resident(&self, mtp: bool) -> Result<()> {
        let mut w = SafeWriter::new();
        self.add_int8(&mut w, "model.embed_tokens.weight")?;
        self.add_int8(&mut w, "lm_head.weight")?;
        w.add_widened(self.src, "model.norm.weight")?;
        for l in (0..self.last).chain(mtp.then_some(self.d.n_layers)) {
            self.add_layer(&mut w, l)?;
        }
        let rpath = format!("{}/resident.safetensors", self.args.out_dir);
        w.write(&rpath)?;
        eprintln!("convert: wrote {rpath}");
        Ok(())
    }

    /// Add a bf16 `[o,i]` tensor requantized to int8 (embed, lm_head).
    fn add_int8(&self, w: &mut SafeWriter<'a>, name: &str) -> Result<()> {
        let (bytes, shape) = self.src.typed(name, Dtype::Bf16)?;
        let (o, i) = (shape[0], shape[1]);
        let (packed, scales) = requant_int8(bytes, o, i);
        w.add(name, Dtype::I8, vec![o, i], packed);
        w.add(format!("{name}.scale"), Dtype::F32, vec![o], scales);
        Ok(())
    }

    /// One layer's share of the resident set: attention always, then whichever MLP half the
    /// layer keeps in memory, plus the MTP head's own four on the one layer that has them.
    fn add_layer(&self, w: &mut SafeWriter<'a>, l: usize) -> Result<()> {
        let lb = format!("model.layers.{l}");
        self.add_attn(w, &lb)?;
        if l < self.d.dense_layers {
            self.add_dense_mlp(w, &lb)?;
        } else {
            self.add_router(w, &lb)?;
        }
        if l == self.d.n_layers {
            self.add_mtp(w, &lb)?;
        }
        Ok(())
    }

    /// The attention block's resident tensors: the four norms widened to f32, the five
    /// projections copied fp8-native.
    fn add_attn(&self, w: &mut SafeWriter<'a>, lb: &str) -> Result<()> {
        w.add_widened(self.src, &format!("{lb}.input_layernorm.weight"))?;
        w.add_widened(self.src, &format!("{lb}.post_attention_layernorm.weight"))?;
        w.add_widened(self.src, &format!("{lb}.self_attn.q_a_layernorm.weight"))?;
        w.add_widened(self.src, &format!("{lb}.self_attn.kv_a_layernorm.weight"))?;
        for p in [
            "q_a_proj",
            "q_b_proj",
            "kv_a_proj_with_mqa",
            "kv_b_proj",
            "o_proj",
        ] {
            w.copy_fp8(self.src, &format!("{lb}.self_attn.{p}"))?;
        }
        Ok(())
    }

    /// A dense layer's whole MLP, copied fp8-native — it has no experts to stream.
    fn add_dense_mlp(&self, w: &mut SafeWriter<'a>, lb: &str) -> Result<()> {
        for p in PROJ {
            w.copy_fp8(self.src, &format!("{lb}.mlp.{p}"))?;
        }
        Ok(())
    }

    /// A MoE layer's router — gate widened to f32 plus its f32 correction bias. All of the
    /// MLP that stays resident: the experts themselves live in the layer's `.vq3`.
    fn add_router(&self, w: &mut SafeWriter<'a>, lb: &str) -> Result<()> {
        w.add_widened(self.src, &format!("{lb}.mlp.gate.weight"))?;
        let bias_name = format!("{lb}.mlp.gate.e_score_correction_bias");
        let (bias, bsh) = self.src.typed(&bias_name, Dtype::F32)?;
        w.add(bias_name, Dtype::F32, bsh.to_vec(), bias.to_vec());
        Ok(())
    }

    /// The MTP-only tensors. `eh_proj` is [hidden, 2·hidden] bf16 in the source with
    /// no block scales, so it cannot ride `copy_fp8`; widened it is 302 MB against a
    /// 15 GB resident set and reuses gemv_f32.
    // ponytail: f32 eh_proj, int8 it if the resident budget ever bites.
    fn add_mtp(&self, w: &mut SafeWriter<'a>, lb: &str) -> Result<()> {
        for t in [
            "enorm.weight",
            "hnorm.weight",
            "eh_proj.weight",
            "shared_head.norm.weight",
        ] {
            w.add_widened(self.src, &format!("{lb}.{t}"))?;
        }
        Ok(())
    }

    /// Stages 4 and 5 — manifest.json (source config + a `format` section), then it and the
    /// tokenizer files, so the artifact is self-contained (the runtime loads tokenizer.json
    /// + generation_config.json from the model dir).
    fn finish(&self, cfg: &serde_json::Value, mtp: bool) -> Result<()> {
        let mut manifest = cfg.clone();
        // A `--layers`-limited convert only writes layers `0..last`, so report the reduced
        // count — the loader iterates `0..num_hidden_layers` and would otherwise demand
        // absent layer files.
        if self.args.layers.is_some() {
            manifest["num_hidden_layers"] = serde_json::json!(self.last);
        }
        manifest["num_nextn_predict_layers"] = serde_json::json!(u8::from(mtp));
        manifest["format"] = serde_json::to_value(FormatMeta::current(FP8_BLOCK))?;
        finish_artifact(
            "convert",
            &self.args.out_dir,
            &self.args.fp8_dir,
            &manifest,
            &["tokenizer.json", "generation_config.json"],
        )
    }
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
    let job = Job {
        args: &args,
        d: &d,
        src: &src,
        last,
    };
    let codebooks = job.codebooks()?;
    let encoders = job.encoders(&codebooks)?;
    let mtp = job.mtp();
    eprintln!(
        "convert: mtp head {}",
        if mtp { "carried" } else { "absent" }
    );
    job.encode_layers(&codebooks, encoders.as_ref(), mtp)?;
    job.write_resident(mtp)?;
    job.finish(&cfg, mtp)?;
    eprintln!("convert: done — {} → {}", args.fp8_dir, args.out_dir);
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use rivoli_artifact::quant::{VqProj, matvec_vq, vq_expert};

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

    /// One projection as the test itself encoded it — the expected side of the write↔read
    /// contract, kept so the loader's slice can be scored against it.
    struct Encoded {
        indices: Vec<u8>,
        scales: Vec<u16>,
        o_dim: usize,
        i_dim: usize,
    }

    /// Three toy codebooks whose only reachable entries are 0 and 1: the rest are 1e30, so
    /// a nearest-lookup can never pick them and the encode's choice stays predictable.
    fn toy_codebooks() -> [Vec<f32>; 3] {
        std::array::from_fn(|p| {
            let mut cb = vec![1e30f32; VQ_K * VQ_DIM];
            cb[0..VQ_DIM].copy_from_slice(&[1.0, -1.0, 0.5 + p as f32, -0.5]);
            cb[VQ_DIM..2 * VQ_DIM].copy_from_slice(&[0.25, 0.5, 0.75, 1.0]);
            cb
        })
    }

    /// Build the expert block directly (no fp8 source): quantize each projection and
    /// `write_proj` it at the offset `Encode::expert` would have used.
    fn build_block(
        cbs: &[Vec<f32>; 3],
        hidden: usize,
        moe_inter: usize,
    ) -> (Vec<u8>, Vec<Encoded>) {
        let mut block = vec![0u8; vq_expert_bytes(hidden, moe_inter)];
        let (mut off, mut originals) = (0usize, Vec::new());
        for (p, &(o_dim, i_dim)) in vq_expert_layout(hidden, moe_inter).iter().enumerate() {
            let mut wv = vec![0.0f32; o_dim * i_dim];
            for (n, x) in wv.iter_mut().enumerate() {
                *x = cbs[p][(n % 2) * VQ_DIM + (n % VQ_DIM)] * 0.5;
            }
            let (indices, scales) = quant_vq(&wv, o_dim, i_dim, &cbs[p]);
            let pb = vq_proj_bytes(o_dim, i_dim);
            write_proj(&mut block[off..off + pb], [o_dim, i_dim], &indices, &scales);
            off += pb;
            originals.push(Encoded {
                indices,
                scales,
                o_dim,
                i_dim,
            });
        }
        (block, originals)
    }

    /// GEMV the loader's slice and the bytes the test wrote and require identical output.
    /// Scored on the values rather than on the bytes, so it fails alike for a wrong offset
    /// and for a scale read at the wrong stride.
    fn assert_gemv_matches(loaded: &VqProj<'_>, want: &Encoded, codebook: &[f32]) {
        let x: Vec<f32> = (0..want.i_dim).map(|v| (v + 1) as f32).collect();
        let (mut yl, mut yr) = (vec![0.0f32; want.o_dim], vec![0.0f32; want.o_dim]);
        let ls = loaded.scales_u16();
        let shape = [loaded.o_dim, loaded.i_dim];
        matvec_vq(&mut yl, &x, VqW::new(loaded.indices, &ls, codebook), shape);
        matvec_vq(
            &mut yr,
            &x,
            VqW::new(&want.indices, &want.scales, codebook),
            [want.o_dim, want.i_dim],
        );
        assert_eq!(yl, yr);
    }

    #[test]
    fn expert_block_slices_via_loader() {
        // Encode a synthetic expert against tiny codebooks, then slice it with the
        // loader's vq_expert and check GEMV reproduces — the write↔read contract.
        let (hidden, moe_inter) = (VQ_GROUP, VQ_GROUP);
        let cbs = toy_codebooks();
        let (block, originals) = build_block(&cbs, hidden, moe_inter);
        let projs = vq_expert(&block, 0, hidden, moe_inter);
        for (k, (loaded, want)) in projs.iter().zip(&originals).enumerate() {
            assert_gemv_matches(loaded, want, &cbs[k]);
        }
    }
}
