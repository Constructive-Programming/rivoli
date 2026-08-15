//! fp8_to_i4 — derive the artifact's `.i4` expert set DIRECTLY from the original fp8
//! checkpoint, replacing the `fp8 → vq3 → int4` chain `bin/vq3_to_i4` produces.
//!
//! `vq3_to_i4` re-quantizes the already-lossy 3-bit `.vq3`, so by construction the
//! int4 set cannot be better than the vq3 it came from (docs/measurement/benchmarks.md, "int4
//! provenance"). This reads the fp8 e4m3 block-scaled experts, dequantizes to f32,
//! and runs the SAME `quant_i4` into the SAME on-disk layout — one quantization step
//! instead of two.
//!
//! IO-BOUND on the source read (~9.7 GiB of fp8 per layer, over NFS here); the actual
//! arithmetic is a dequant and a per-group amax. A GPU port would buy nothing — unlike
//! the vq3 encoder, whose codebook argmin is genuinely compute-heavy.
//!
//! The `.i4` set is ~365 GB and does not fit twice on disk, so layers are replaced IN
//! PLACE: write `L{l}.i4.tmp`, fsync, `rename(2)` over `L{l}.i4`. Peak extra usage is
//! one layer; free space is checked before each. Overwriting is NOT recoverable in
//! place: `bin/vq3_to_i4`, which used to regenerate the old `.i4` from the `.vq3` set,
//! is deleted (its output was worse by construction — see the paragraph above). Re-run
//! THIS tool from the fp8 source to rebuild a layer.
use anyhow::{Context, Result, anyhow, ensure};
use clap::Parser;
use rivoli_artifact::{
    format::{FormatMeta, I4Source, Safetensors},
    glm_config::ModelConfig,
    quant::{
        ExpertProjs, I4_GROUP, RowScaledW, expert_base, expert_projs, i4_expert_bytes,
        i4_expert_stride, i4_slot_offsets, quant_i4, write_i4_proj,
    },
};
use std::fs::File;
use std::io::Write;

// NOTE: doc comments on the FIELDS below are USER-FACING — clap renders them as `--help`.
// Rationale for the code goes in `//` comments like this one, which clap ignores. The
// `USAGE` const this replaced was the only place `--to`'s exclusivity was written down,
// and it was a string the compiler could not check against the loop beneath it.
#[derive(Parser)]
#[command(
    name = "fp8_to_i4",
    about = "Derive an artifact's .i4 expert set directly from the original fp8 checkpoint"
)]
struct Args {
    /// The fp8 GLM-5.2 checkpoint to quantize from. Its tile size must match the one the
    /// artifact's manifest records, so a wrong-model or wrong-revision checkpoint dies on
    /// the first expert instead of writing plausible garbage.
    fp8_dir: String,

    /// The artifact directory whose `L{ll}.i4` files are written. Layers are replaced IN
    /// PLACE (tmp, fsync, rename), so peak extra space is one layer — and the overwrite is
    /// NOT recoverable in place: re-run this tool against the fp8 source to rebuild one.
    artifact_dir: String,

    /// First layer to convert (inclusive). Defaults to the artifact's first MoE layer;
    /// this is the flag an aborted run tells you to resume with.
    #[arg(long, value_name = "L")]
    from: Option<usize>,

    /// One PAST the last layer to convert — `--to` is exclusive. Defaults to one past the
    /// MTP head when the checkpoint carries one, else `num_hidden_layers`.
    #[arg(long, value_name = "L")]
    to: Option<usize>,
}

/// Bytes as GiB: every size this tool reports is a layer or a filesystem, and neither is
/// legible in bytes.
fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

/// Everything that is the same for every layer and every expert — the fp8 source, the `.i4`
/// slot layout, the worker split. Hoisted out of the per-expert worker loop for the reason
/// it always was: the name/shape pairing is the same for every expert of every layer. As
/// one value it also keeps the per-expert call at `(base, slot)` instead of the
/// six-argument list it had been.
struct Encoder<'a> {
    src: &'a Safetensors,
    /// The fp8 `weight_scale_inv` tile size the artifact's manifest records — checked
    /// against every projection by `dequant_fp8`, see [`main`].
    block: usize,
    projs: ExpertProjs,
    off: [usize; 6],
    n_experts: usize,
    /// Expert blocks per layer: routed `0..n_experts`, then the shared expert.
    blocks: usize,
    stride: usize,
    /// One expert's UNPADDED bytes. The gap up to `stride` is left as the caller found it,
    /// which is zero — see the single zeroed allocation in [`main`].
    ebytes: usize,
    /// Experts per worker thread.
    per: usize,
    /// fp8 source bytes per expert: three projections of `hidden × moe_inter` at one byte
    /// per weight. This tool is IO-bound on that read, so it is the numerator of the
    /// per-layer rate line rather than the `.i4` bytes written.
    fp8_bytes: usize,
}

impl<'a> Encoder<'a> {
    fn new(cfg: &ModelConfig, src: &'a Safetensors, block: usize) -> Self {
        let (h, m, ne) = (cfg.hidden, cfg.moe_inter, cfg.n_experts);
        let blocks = ne + 1;
        let threads = std::thread::available_parallelism().map_or(8, |t| t.get());
        Encoder {
            src,
            block,
            projs: expert_projs(h, m),
            off: i4_slot_offsets(h, m),
            n_experts: ne,
            blocks,
            stride: i4_expert_stride(h, m),
            ebytes: i4_expert_bytes(h, m),
            per: blocks.div_ceil(threads),
            fp8_bytes: 3 * h * m,
        }
    }

    /// On-disk (and in-buffer) bytes of one layer: every expert block at its full stride.
    fn layer_bytes(&self) -> u64 {
        (self.blocks * self.stride) as u64
    }

    /// Workers actually spawned. `per` is a ceiling, so the last chunk can be short and this
    /// can come out below `available_parallelism` — which is the number worth reporting.
    fn workers(&self) -> usize {
        self.blocks.div_ceil(self.per)
    }

    /// Quantize one expert (routed or shared) rooted at `base` straight from fp8 into
    /// `slot`, gate‖up‖down at the offsets `i4_slot_offsets` defines. `slot` is exactly
    /// one expert's unpadded bytes; the caller leaves the stride padding zero.
    fn build_block(&self, base: &str, slot: &mut [u8]) -> Result<()> {
        for (k, &(proj, (o_dim, i_dim))) in self.projs.iter().enumerate() {
            let name = format!("{base}.{proj}");
            let w = self.src.dequant_fp8(&name, [o_dim, i_dim], self.block)?;
            let (packed, scale) = quant_i4(&w, o_dim, i_dim);
            write_i4_proj(slot, &self.off, k, RowScaledW::new(&packed, &scale));
        }
        Ok(())
    }

    /// One worker's share of layer `l`: experts `ci·per ..`, one per `stride`-sized slot of
    /// `chunk`. The error names the expert, since a checkpoint mismatch dies on the first.
    fn fill_chunk(&self, chunk: &mut [u8], l: usize, ci: usize) -> Result<()> {
        for (j, slot) in chunk.chunks_exact_mut(self.stride).enumerate() {
            let e = ci * self.per + j;
            let base = expert_base(l, e, self.n_experts);
            self.build_block(&base, &mut slot[..self.ebytes])
                .with_context(|| format!("expert {e}"))?;
        }
        Ok(())
    }

    /// Fill `buf` with every expert block of layer `l`, `per` experts to a worker. The split
    /// is by disjoint `chunks_mut` slices, so the borrow checker witnesses that no two
    /// workers can touch the same block.
    fn convert_layer(&self, buf: &mut [u8], l: usize) -> Result<()> {
        std::thread::scope(|s| -> Result<()> {
            let handles: Vec<_> = buf
                .chunks_mut(self.per * self.stride)
                .enumerate()
                .map(|(ci, chunk)| s.spawn(move || self.fill_chunk(chunk, l, ci)))
                .collect();
            for hd in handles {
                hd.join().map_err(|_| anyhow!("encode worker panicked"))??;
            }
            Ok(())
        })
        .with_context(|| format!("convert layer {l}"))
    }
}

/// Bytes available to an unprivileged writer on the filesystem holding `dir`.
fn free_bytes(dir: &str) -> Result<u64> {
    let c = std::ffi::CString::new(dir).context("path contains NUL")?;
    // SAFETY: `statvfs` is a POD of unsigned integers, so all-zero is a valid value
    // to hand the kernel; `c` is a valid NUL-terminated path. Fields read only after
    // the `rc == 0` check below.
    let mut s: libc::statvfs = unsafe { std::mem::zeroed() };
    let rc = unsafe { libc::statvfs(c.as_ptr(), &mut s) };
    ensure!(
        rc == 0,
        "statvfs({dir}): {}",
        std::io::Error::last_os_error()
    );
    Ok(s.f_bavail as u64 * s.f_frsize as u64)
}

/// The `from..to` range to convert: the two flags defaulted against the artifact and the
/// checkpoint, then checked against the MoE range.
fn layer_range(
    cfg: &ModelConfig,
    src: &Safetensors,
    arg_from: Option<usize>,
    arg_to: Option<usize>,
) -> Result<(usize, usize)> {
    // The MTP head is checkpoint layer `n_layers` — a full MoE layer, so it converts
    // through this exact loop with no special case. It was excluded only because the
    // bound below read `<= cfg.n_layers`, and that one comparison is why `--mode int4`
    // and the default `--mode hybrid` carried no head and silently decoded sequentially
    // ("speculative decode OFF: this artifact carries no MTP head"). `convert` detects it
    // the same way.
    let mtp = src.has(&format!("model.layers.{}.eh_proj.weight", cfg.n_layers));
    let last = cfg.n_layers + usize::from(mtp);
    let (from, to) = (arg_from.unwrap_or(cfg.dense_layers), arg_to.unwrap_or(last));
    ensure!(
        cfg.dense_layers <= from && from < to && to <= last,
        "layer range {from}..{to} outside the MoE range {}..{last}",
        cfg.dense_layers,
    );
    Ok((from, to))
}

/// Refuse to start a layer that could not complete — a full `/` is far worse than an
/// unfinished run, and the message names the flag that resumes.
fn ensure_space(art: &str, layer_bytes: u64, l: usize) -> Result<()> {
    // Peak extra usage is one layer: the tmp exists alongside the file it
    // replaces, and the old blocks free at the rename.
    let (free, need) = (free_bytes(art)?, layer_bytes + (1 << 30));
    ensure!(
        free >= need,
        "only {:.1} GiB free on {art}, need {:.1} GiB for L{l:02}.i4 — aborting before the \
         disk fills (resume with --from {l})",
        gib(free),
        gib(need)
    );
    Ok(())
}

/// Publish `buf` at `path` the way the module header describes: write `tmp`, fsync, then
/// `rename(2)` over the file being replaced, so a killed run never leaves a torn `.i4`.
fn publish(tmp: &str, path: &str, buf: &[u8]) -> Result<()> {
    let mut f = File::create(tmp).with_context(|| format!("create {tmp}"))?;
    f.write_all(buf).with_context(|| format!("write {tmp}"))?;
    f.sync_all().with_context(|| format!("fsync {tmp}"))?;
    drop(f);
    std::fs::rename(tmp, path).with_context(|| format!("rename {tmp} -> {path}"))
}

/// Convert layer `l` from fp8 into `buf` and publish it over `{art}/L{ll}.i4`.
fn write_layer(enc: &Encoder<'_>, art: &str, buf: &mut [u8], l: usize) -> Result<()> {
    // The tmp name carries the pid for the reason `format::write_expert_layer` gives at
    // length: agents share this machine and a convert takes no lock, so two runs into one
    // artifact dir are reachable. On a FIXED `L{ll}.i4.tmp` both would `File::create` +
    // truncate + write concurrently, and the rename would publish interleaved bytes —
    // which, being two writes OF EQUAL LENGTH, yields a file of exactly the right length
    // and sails through `open_routed`'s length check. This was the last fixed-name
    // publisher in the tree (found by review 2026-08-10).
    let path = format!("{art}/L{l:02}.i4");
    let tmp = format!("{path}.{}.tmp", std::process::id());
    // Drop a tmp left by an aborted run of THIS pid before measuring free space — it is
    // the space this layer is about to reuse, and counting it would refuse the very
    // resume the abort message recommends. Another pid's tmp is not ours to remove, and
    // is now distinguishable.
    let _ = std::fs::remove_file(&tmp);
    ensure_space(art, enc.layer_bytes(), l)?;

    let t = std::time::Instant::now();
    enc.convert_layer(buf, l)?;
    publish(&tmp, &path, buf)?;
    let secs = t.elapsed().as_secs_f64();
    eprintln!(
        "  L{l:02}.i4 <- fp8 in {secs:.0}s ({:.0} MiB/s fp8 read)",
        (enc.blocks * enc.fp8_bytes) as f64 / secs / (1u64 << 20) as f64
    );
    Ok(())
}

/// Record exactly what was rebuilt. A partial run stamps its own range (merging
/// with an adjoining earlier run), so a resumed conversion still ends up claiming
/// the whole set and a genuinely mixed artifact never claims to be uniform.
fn stamp_source(art: &str, fp8_dir: &str, range: [usize; 2]) -> Result<()> {
    I4Source {
        tool: "fp8_to_i4".into(),
        chain: "fp8->int4".into(),
        src: std::fs::canonicalize(fp8_dir)
            .with_context(|| format!("canonicalize {fp8_dir}"))?
            .display()
            .to_string(),
        layers: range,
        group: Some(I4_GROUP),
    }
    .stamp(art)
}

/// The merge in [`stamp_source`] only fires against a PRIOR stamp, and a stamp is one JSON
/// field that can go missing (this artifact's did — docs/investigations/int4-scales.md §Reproduction). Converting a
/// subrange into an unstamped set then writes a claim NARROWER than the `.i4` on disk,
/// which reads as "only these layers are fp8-derived" and is worse than no claim at all.
/// Cost 2026-07-31: a --from 78 run stamped [78,79] over a full set and
/// `moe_i4_real_data_vs_fp8_ground_truth` — which had been skipping on the absent stamp
/// — started failing on layer 3. Say so; do not guess the range on the user's behalf.
fn warn_if_stamp_understates(art: &str, cfg: &ModelConfig, from: usize, to: usize) {
    let on_disk: Vec<usize> = (cfg.dense_layers..=cfg.n_layers)
        .filter(|l| std::fs::metadata(format!("{art}/L{l:02}.i4")).is_ok())
        .collect();
    if let (Some(&lo), Some(&hi)) = (on_disk.first(), on_disk.last())
        && (lo < from || hi >= to)
    {
        eprintln!(
            "fp8_to_i4: WARNING — stamped [{from},{to}) but L{lo:02}.i4..L{hi:02}.i4 are on \
             disk. The stamp now UNDERSTATES the set. If those layers came from this same \
             fp8 source at group {I4_GROUP}, widen `i4_source.layers` to [{lo},{}].",
            hi + 1
        );
    }
}

fn main() -> Result<()> {
    let Args {
        fp8_dir,
        artifact_dir: art,
        from: arg_from,
        to: arg_to,
    } = Args::parse();

    let cfg = ModelConfig::load(&art).context("load artifact manifest")?;
    // The artifact records the fp8 tile size it was built with. Every projection's
    // `weight_scale_inv` extent is then checked against it in `dequant_fp8`, so a
    // checkpoint that disagrees — wrong dims, wrong tiling, wrong model — fails hard
    // on the first expert rather than producing plausible-looking garbage.
    let block = FormatMeta::load(&art)?.fp8_block;

    let src = Safetensors::open_dir(&fp8_dir).context("open fp8 checkpoint")?;
    let (from, to) = layer_range(&cfg, &src, arg_from, arg_to)?;

    let enc = Encoder::new(&cfg, &src, block);
    eprintln!(
        "fp8_to_i4: layers {from}..{to}, {} blocks/layer, {:.2} GiB/layer, {} workers",
        enc.blocks,
        gib(enc.layer_bytes()),
        enc.workers()
    );

    // One buffer for every layer: each pass fully overwrites all `n` slots, and the
    // stride padding stays zero from this single zeroed allocation.
    let mut buf = vec![0u8; enc.blocks * enc.stride];
    for l in from..to {
        write_layer(&enc, &art, &mut buf, l)?;
    }

    stamp_source(&art, &fp8_dir, [from, to])?;
    warn_if_stamp_understates(&art, &cfg, from, to);
    eprintln!(
        "fp8_to_i4: done — layers {from}..{to} rebuilt from fp8 at group {I4_GROUP}, manifest i4_source stamped"
    );
    Ok(())
}
