//! fp8_to_i4 — derive the artifact's `.i4` expert set DIRECTLY from the original fp8
//! checkpoint, replacing the `fp8 → vq3 → int4` chain `bin/vq3_to_i4` produces.
//!
//! `vq3_to_i4` re-quantizes the already-lossy 3-bit `.vq3`, so by construction the
//! int4 set cannot be better than the vq3 it came from (benchmarks.md, "int4
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
//! one layer; free space is checked before each. Overwriting is recoverable — the
//! `.vq3` set is still present and `bin/vq3_to_i4` regenerates the old `.i4` from it.
//!
//! usage: fp8_to_i4 <fp8-dir> <artifact-dir> [--from L] [--to L]   (`--to` exclusive)
use anyhow::{Context, Result, anyhow, ensure};
use rivoli::format::{FormatMeta, I4Source, Safetensors};
use rivoli::model::ModelConfig;
use rivoli::quant::{
    I4_GROUP, i4_expert_bytes, i4_expert_stride, i4_slot_offsets, quant_i4, vq_expert_layout,
    write_i4_proj,
};
use std::fs::File;
use std::io::Write;

const PROJ: [&str; 3] = ["gate_proj", "up_proj", "down_proj"];

/// Quantize one expert (routed or shared) rooted at `base` straight from fp8 into
/// `slot`, gate‖up‖down at the offsets `i4_slot_offsets` defines. `slot` is exactly
/// one expert's unpadded bytes; the caller leaves the stride padding zero.
fn build_block(
    src: &Safetensors,
    base: &str,
    hidden: usize,
    moe_inter: usize,
    block: usize,
    off: &[usize; 6],
    slot: &mut [u8],
) -> Result<()> {
    for (k, (proj, &(o_dim, i_dim))) in PROJ
        .iter()
        .zip(&vq_expert_layout(hidden, moe_inter))
        .enumerate()
    {
        let w = src.dequant_fp8(&format!("{base}.{proj}"), o_dim, i_dim, block)?;
        let (packed, scale) = quant_i4(&w, o_dim, i_dim);
        write_i4_proj(slot, off, k, &packed, &scale);
    }
    Ok(())
}

/// Bytes available to an unprivileged writer on the filesystem holding `dir`.
fn free_bytes(dir: &str) -> Result<u64> {
    let c = std::ffi::CString::new(dir).context("path contains NUL")?;
    // SAFETY: `statvfs` is a POD of unsigned integers, so all-zero is a valid value
    // to hand the kernel; `c` is a valid NUL-terminated path. Fields read only after
    // the `rc == 0` check below.
    let mut s: libc::statvfs = unsafe { std::mem::zeroed() };
    let rc = unsafe { libc::statvfs(c.as_ptr(), &mut s) };
    ensure!(rc == 0, "statvfs({dir}): {}", std::io::Error::last_os_error());
    Ok(s.f_bavail as u64 * s.f_frsize as u64)
}

fn main() -> Result<()> {
    const USAGE: &str = "usage: fp8_to_i4 <fp8-dir> <artifact-dir> [--from L] [--to L]   (--to exclusive)";
    let (mut arg_from, mut arg_to, mut pos) = (None, None, Vec::new());
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--from" => arg_from = Some(it.next().context("--from L")?.parse()?),
            "--to" => arg_to = Some(it.next().context("--to L")?.parse()?),
            _ => pos.push(a),
        }
    }
    let [fp8_dir, art]: [String; 2] = pos.try_into().map_err(|_| anyhow!(USAGE))?;

    let cfg = ModelConfig::load(&art).context("load artifact manifest")?;
    let (h, m, ne) = (cfg.hidden, cfg.moe_inter, cfg.n_experts);
    // The artifact records the fp8 tile size it was built with. Every projection's
    // `weight_scale_inv` extent is then checked against it in `dequant_fp8`, so a
    // checkpoint that disagrees — wrong dims, wrong tiling, wrong model — fails hard
    // on the first expert rather than producing plausible-looking garbage.
    let block = FormatMeta::load(&art)?.fp8_block;

    let (from, to) = (
        arg_from.unwrap_or(cfg.dense_layers),
        arg_to.unwrap_or(cfg.n_layers),
    );
    ensure!(
        cfg.dense_layers <= from && from < to && to <= cfg.n_layers,
        "layer range {from}..{to} outside the MoE range {}..{}",
        cfg.dense_layers,
        cfg.n_layers
    );

    let (stride, ebytes) = (i4_expert_stride(h, m), i4_expert_bytes(h, m));
    let off = i4_slot_offsets(h, m);
    let n = ne + 1; // routed 0..ne, then the shared expert
    let layer_bytes = (n * stride) as u64;
    let threads = std::thread::available_parallelism().map_or(8, |t| t.get());
    let per = n.div_ceil(threads);
    let src = Safetensors::open_dir(&fp8_dir).context("open fp8 checkpoint")?;
    eprintln!(
        "fp8_to_i4: layers {from}..{to}, {n} blocks/layer, {:.2} GiB/layer, {} workers",
        layer_bytes as f64 / (1u64 << 30) as f64,
        n.div_ceil(per)
    );

    // One buffer for every layer: each pass fully overwrites all `n` slots, and the
    // stride padding stays zero from this single zeroed allocation.
    let mut buf = vec![0u8; n * stride];
    for l in from..to {
        let (path, tmp) = (format!("{art}/L{l:02}.i4"), format!("{art}/L{l:02}.i4.tmp"));
        // Drop a tmp left by an aborted run BEFORE measuring free space — it is the
        // space this layer is about to reuse, and counting it would refuse the very
        // resume the abort message recommends.
        let _ = std::fs::remove_file(&tmp);
        // Peak extra usage is one layer: the tmp exists alongside the file it
        // replaces, and the old blocks free at the rename. Refuse to start a layer
        // that could not complete — a full `/` is far worse than an unfinished run.
        let (free, need) = (free_bytes(&art)?, layer_bytes + (1 << 30));
        ensure!(
            free >= need,
            "only {:.1} GiB free on {art}, need {:.1} GiB for L{l:02}.i4 — aborting before the \
             disk fills (resume with --from {l})",
            free as f64 / (1u64 << 30) as f64,
            need as f64 / (1u64 << 30) as f64
        );

        let t = std::time::Instant::now();
        std::thread::scope(|s| -> Result<()> {
            let handles: Vec<_> = buf
                .chunks_mut(per * stride)
                .enumerate()
                .map(|(ci, chunk)| {
                    let (src, off) = (&src, &off);
                    s.spawn(move || -> Result<()> {
                        for (j, slot) in chunk.chunks_exact_mut(stride).enumerate() {
                            let e = ci * per + j;
                            let base = if e < ne {
                                format!("model.layers.{l}.mlp.experts.{e}")
                            } else {
                                format!("model.layers.{l}.mlp.shared_experts")
                            };
                            build_block(src, &base, h, m, block, off, &mut slot[..ebytes])
                                .with_context(|| format!("expert {e}"))?;
                        }
                        Ok(())
                    })
                })
                .collect();
            for hd in handles {
                hd.join().map_err(|_| anyhow!("encode worker panicked"))??;
            }
            Ok(())
        })
        .with_context(|| format!("convert layer {l}"))?;

        let mut f = File::create(&tmp).with_context(|| format!("create {tmp}"))?;
        f.write_all(&buf).with_context(|| format!("write {tmp}"))?;
        f.sync_all().with_context(|| format!("fsync {tmp}"))?;
        drop(f);
        std::fs::rename(&tmp, &path).with_context(|| format!("rename {tmp} -> {path}"))?;
        let secs = t.elapsed().as_secs_f64();
        eprintln!(
            "  L{l:02}.i4 <- fp8 in {secs:.0}s ({:.0} MiB/s fp8 read)",
            (n as f64 * 3.0 * (h * m) as f64) / secs / (1u64 << 20) as f64
        );
    }

    // Record exactly what was rebuilt. A partial run stamps its own range (merging
    // with an adjoining earlier run), so a resumed conversion still ends up claiming
    // the whole set and a genuinely mixed artifact never claims to be uniform.
    I4Source {
        tool: "fp8_to_i4".into(),
        chain: "fp8->int4".into(),
        src: std::fs::canonicalize(&fp8_dir)
            .with_context(|| format!("canonicalize {fp8_dir}"))?
            .display()
            .to_string(),
        layers: [from, to],
        group: Some(I4_GROUP),
    }
    .stamp(&art)?;
    eprintln!(
        "fp8_to_i4: done — layers {from}..{to} rebuilt from fp8 at group {I4_GROUP}, manifest i4_source stamped"
    );
    Ok(())
}
