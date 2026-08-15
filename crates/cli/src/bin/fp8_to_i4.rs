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
        ExpertProjs, I4_GROUP, expert_base, expert_projs, i4_expert_bytes, i4_expert_stride,
        i4_slot_offsets, quant_i4, write_i4_proj,
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

/// Quantize one expert (routed or shared) rooted at `base` straight from fp8 into
/// `slot`, gate‖up‖down at the offsets `i4_slot_offsets` defines. `slot` is exactly
/// one expert's unpadded bytes; the caller leaves the stride padding zero.
fn build_block(
    src: &Safetensors,
    base: &str,
    projs: &ExpertProjs,
    block: usize,
    off: &[usize; 6],
    slot: &mut [u8],
) -> Result<()> {
    for (k, &(proj, (o_dim, i_dim))) in projs.iter().enumerate() {
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
    ensure!(
        rc == 0,
        "statvfs({dir}): {}",
        std::io::Error::last_os_error()
    );
    Ok(s.f_bavail as u64 * s.f_frsize as u64)
}

fn main() -> Result<()> {
    let Args {
        fp8_dir,
        artifact_dir: art,
        from: arg_from,
        to: arg_to,
    } = Args::parse();

    let cfg = ModelConfig::load(&art).context("load artifact manifest")?;
    let (h, m, ne) = (cfg.hidden, cfg.moe_inter, cfg.n_experts);
    // The artifact records the fp8 tile size it was built with. Every projection's
    // `weight_scale_inv` extent is then checked against it in `dequant_fp8`, so a
    // checkpoint that disagrees — wrong dims, wrong tiling, wrong model — fails hard
    // on the first expert rather than producing plausible-looking garbage.
    let block = FormatMeta::load(&art)?.fp8_block;

    let src = Safetensors::open_dir(&fp8_dir).context("open fp8 checkpoint")?;

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

    let (stride, ebytes) = (i4_expert_stride(h, m), i4_expert_bytes(h, m));
    let off = i4_slot_offsets(h, m);
    // Hoisted out of the per-expert worker loop: the name/shape pairing is the same for
    // every expert of every layer.
    let projs = expert_projs(h, m);
    let n = ne + 1; // routed 0..ne, then the shared expert
    let layer_bytes = (n * stride) as u64;
    let threads = std::thread::available_parallelism().map_or(8, |t| t.get());
    let per = n.div_ceil(threads);
    eprintln!(
        "fp8_to_i4: layers {from}..{to}, {n} blocks/layer, {:.2} GiB/layer, {} workers",
        layer_bytes as f64 / (1u64 << 30) as f64,
        n.div_ceil(per)
    );

    // One buffer for every layer: each pass fully overwrites all `n` slots, and the
    // stride padding stays zero from this single zeroed allocation.
    let mut buf = vec![0u8; n * stride];
    for l in from..to {
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
                    let (src, off, projs) = (&src, &off, &projs);
                    s.spawn(move || -> Result<()> {
                        for (j, slot) in chunk.chunks_exact_mut(stride).enumerate() {
                            let e = ci * per + j;
                            let base = expert_base(l, e, ne);
                            build_block(src, &base, projs, block, off, &mut slot[..ebytes])
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
    // The merge above only fires against a PRIOR stamp, and a stamp is one JSON field that
    // can go missing (this artifact's did — docs/investigations/int4-scales.md §Reproduction). Converting a
    // subrange into an unstamped set then writes a claim NARROWER than the `.i4` on disk,
    // which reads as "only these layers are fp8-derived" and is worse than no claim at all.
    // Cost 2026-07-31: a --from 78 run stamped [78,79] over a full set and
    // `moe_i4_real_data_vs_fp8_ground_truth` — which had been skipping on the absent stamp
    // — started failing on layer 3. Say so; do not guess the range on the user's behalf.
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
    eprintln!(
        "fp8_to_i4: done — layers {from}..{to} rebuilt from fp8 at group {I4_GROUP}, manifest i4_source stamped"
    );
    Ok(())
}
