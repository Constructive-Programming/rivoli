//! Repack colibri's per-expert int4 tensors into our per-layer streaming `.i4`
//! files (one aligned block per expert = gate‖gate_scale‖up‖up_scale‖down‖
//! down_scale, index `n_experts` = the shared expert), written BESIDE the `.vq3`
//! files in the artifact. colibri's int4 layout (per-row, low-nibble-first,
//! `(nibble-8)·qs`) is byte-identical to ours, so this is a pure copy — no re-quant.
//!
//! usage: pack_i4 <colibri-dir> <artifact-dir> [--layers N] [--experts M]
//!   --layers N   convert only the first N MoE layers (smoke test)
//!   --experts M  convert only experts 0..M plus the shared expert (smoke test)
use anyhow::{Context, Result, bail, ensure};
use rivoli::model::ModelConfig;
use rivoli::quant::{i4_expert_stride, i4_row_bytes, i4_slot_offsets};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;

/// Location of one tensor: (shard path, absolute byte start, length).
struct Shards {
    loc: HashMap<String, (PathBuf, u64, usize)>,
}

impl Shards {
    /// Scan every `out-*.safetensors` header once, recording each tensor's location.
    fn scan(dir: &str) -> Result<Self> {
        let mut loc = HashMap::new();
        let mut shards: Vec<_> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("out-") && n.ends_with(".safetensors"))
            })
            .collect();
        shards.sort();
        ensure!(!shards.is_empty(), "no out-*.safetensors in {dir}");
        for p in shards {
            let mut f = File::open(&p)?;
            let mut lenb = [0u8; 8];
            f.read_exact(&mut lenb)?;
            let n = u64::from_le_bytes(lenb) as usize;
            let mut hdr = vec![0u8; n];
            f.read_exact(&mut hdr)?;
            let j: serde_json::Value = serde_json::from_slice(&hdr)?;
            let base = 8 + n as u64;
            for (name, v) in j.as_object().context("header not an object")? {
                if name == "__metadata__" {
                    continue;
                }
                if let Some((s, e)) = v
                    .get("data_offsets")
                    .and_then(|o| o.as_array())
                    .and_then(|off| Some((off.first()?.as_u64()?, off.get(1)?.as_u64()?)))
                {
                    loc.insert(name.clone(), (p.clone(), base + s, (e - s) as usize));
                }
            }
        }
        Ok(Self { loc })
    }

    fn read(&self, name: &str) -> Result<Vec<u8>> {
        let (p, start, len) = self.loc.get(name).with_context(|| format!("missing {name}"))?;
        let mut f = File::open(p)?;
        f.seek(SeekFrom::Start(*start))?;
        let mut buf = vec![0u8; *len];
        f.read_exact(&mut buf)?;
        Ok(buf)
    }
}

/// Write one expert's block into `blk` (zeroed, `i4_expert_stride` bytes) from the
/// six colibri tensors under `prefix` (`...experts.5` or `...shared_experts`).
fn fill_block(
    sh: &Shards,
    prefix: &str,
    blk: &mut [u8],
    off: &[usize; 6],
    hidden: usize,
    inter: usize,
) -> Result<()> {
    // (dest offset index, tensor suffix, expected bytes)
    let specs = [
        (0usize, "gate_proj.weight", inter * i4_row_bytes(hidden)),
        (1, "gate_proj.weight.qs", inter * 4),
        (2, "up_proj.weight", inter * i4_row_bytes(hidden)),
        (3, "up_proj.weight.qs", inter * 4),
        (4, "down_proj.weight", hidden * i4_row_bytes(inter)),
        (5, "down_proj.weight.qs", hidden * 4),
    ];
    for (oi, suf, want) in specs {
        let bytes = sh.read(&format!("{prefix}.{suf}"))?;
        ensure!(
            bytes.len() == want,
            "{prefix}.{suf}: {} bytes, expected {want}",
            bytes.len()
        );
        blk[off[oi]..off[oi] + want].copy_from_slice(&bytes);
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        bail!("usage: pack_i4 <colibri-dir> <artifact-dir> [--layers N] [--experts M]");
    }
    let (colibri, artifact) = (&args[1], &args[2]);
    let flag = |name: &str| -> Option<usize> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .and_then(|s| s.parse().ok())
    };
    let cfg = ModelConfig::load(artifact).context("load artifact manifest")?;
    let (hidden, inter, ne) = (cfg.hidden, cfg.moe_inter, cfg.n_experts);
    let stride = i4_expert_stride(hidden, inter);
    let off = i4_slot_offsets(hidden, inter);
    let last_layer = cfg.dense_layers + flag("--layers").unwrap_or(cfg.n_layers - cfg.dense_layers);
    let n_exp = flag("--experts").unwrap_or(ne); // routed experts to emit (+ shared)

    eprintln!("scanning colibri shards…");
    let sh = Shards::scan(colibri)?;
    eprintln!(
        "packing layers {}..{last_layer}, {n_exp}+1 experts/layer, stride {:.2} MiB",
        cfg.dense_layers,
        stride as f64 / (1 << 20) as f64
    );
    for l in cfg.dense_layers..last_layer {
        let path = format!("{artifact}/L{l:02}.i4");
        let mut out = File::create(&path).with_context(|| format!("create {path}"))?;
        let mut blk = vec![0u8; stride];
        for e in 0..n_exp {
            blk.iter_mut().for_each(|b| *b = 0);
            let prefix = format!("model.layers.{l}.mlp.experts.{e}");
            fill_block(&sh, &prefix, &mut blk, &off, hidden, inter)?;
            out.write_all(&blk)?;
        }
        // Shared expert at block index `ne` (loader reads it there).
        blk.iter_mut().for_each(|b| *b = 0);
        fill_block(&sh, &format!("model.layers.{l}.mlp.shared_experts"), &mut blk, &off, hidden, inter)?;
        out.write_all(&blk)?;
        eprintln!("  L{l:02}.i4 written ({} blocks)", n_exp + 1);
    }
    eprintln!("done");
    Ok(())
}
