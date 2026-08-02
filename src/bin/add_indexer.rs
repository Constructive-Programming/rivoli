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

use anyhow::{Result, ensure};
use clap::Parser;
use rivoli::artifact::format::{SafeWriter, Safetensors};
use rivoli::artifact::model::ModelConfig;

// NOTE: doc comments on the FIELDS below are USER-FACING — clap renders them as `--help`.
// Rationale for the code goes in `//` comments like this one, which clap ignores. The
// hand-rolled loop this replaced repeated one `usage:` string per positional and, being an
// iterator, silently accepted (and dropped) any third argument.
#[derive(Parser)]
#[command(
    name = "add_indexer",
    about = "Splice DSA lightning-indexer weights from a stash into an existing artifact"
)]
struct Args {
    /// The artifact directory to splice into. `indexer.safetensors` is written beside the
    /// resident set, and `Pin::build` merges every `*.safetensors` in the directory, so
    /// `--attn dsa`/`misa` auto-detect it on the next run. Existing experts are untouched.
    artifact_dir: String,

    /// The stashed safetensors holding every `*.indexer.*` tensor of the fp8 source. Must
    /// cover every FULL indexer layer the manifest declares; a gap is a truncated stash
    /// and fails hard rather than writing a half-usable indexer.
    stash: String,
}

fn main() -> Result<()> {
    let Args {
        artifact_dir: art,
        stash,
    } = Args::parse();

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
