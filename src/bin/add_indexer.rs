//! add_indexer — splice DSA lightning-indexer weights into an existing artifact.
//!
//! The main converter builds a DENSE artifact (no indexer). When the fp8 source is
//! gone but the indexer tensors were stashed (see the cleanup that extracted every
//! `*.indexer.*` tensor), this writes `<artifact>/indexer.safetensors` from the
//! stash so `Pin::build` (which opens the artifact dir, merging every *.safetensors)
//! resolves the resident indexer for dsa/misa — without re-encoding the 276 GB of
//! experts or re-downloading the 581 GB source.
//!
//! Per full layer: `wk`/`wq_b` are copied verbatim (fp8 + weight_scale_inv);
//! `weights_proj` and `k_norm.{weight,bias}` are widened bf16→f32 (the loader reads
//! weights_proj via gemv_f32 and k_norm via the layernorm kernel).
//!
//! usage: add_indexer <artifact-dir> <indexer-stash.safetensors>

use anyhow::{Context, Result, ensure};
use rivoli::format::{SafeWriter, Safetensors};
use rivoli::model::ModelConfig;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let art = args
        .next()
        .context("usage: add_indexer <artifact-dir> <indexer-stash.safetensors>")?;
    let stash = args
        .next()
        .context("usage: add_indexer <artifact-dir> <indexer-stash.safetensors>")?;

    let cfg = ModelConfig::load(&art)?;
    let full = cfg.indexer_layout()?; // validates index dims + per-layer full/shared
    let src = Safetensors::open_file(&stash)?;

    let mut w = SafeWriter::new();
    let mut n_full = 0usize;
    for (l, &is_full) in full.iter().enumerate() {
        if !is_full {
            continue;
        }
        let base = format!("model.layers.{l}.self_attn.indexer");
        // Every full layer's indexer must be in the stash; a gap is a truncated stash.
        ensure!(
            src.has(&format!("{base}.wk.weight")),
            "stash missing indexer weights for full layer {l}"
        );
        w.copy_fp8(&src, &format!("{base}.wk"))?;
        w.copy_fp8(&src, &format!("{base}.wq_b"))?;
        w.add_widened(&src, &format!("{base}.weights_proj.weight"))?;
        w.add_widened(&src, &format!("{base}.k_norm.weight"))?;
        w.add_widened(&src, &format!("{base}.k_norm.bias"))?;
        n_full += 1;
    }
    let out = format!("{art}/indexer.safetensors");
    w.write(&out)?;
    eprintln!("add_indexer: wrote {out} ({n_full} full layers)");
    Ok(())
}
