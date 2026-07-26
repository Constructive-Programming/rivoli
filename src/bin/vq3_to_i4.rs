//! Re-derive the `.i4` experts from our OWN faithful `.vq3` weights (glm52-fp8 source)
//! instead of copying colibri's int4. Colibri int4 is a measurably worse/mismatched
//! quantization of the experts (per-row RTN R~0.96 vs vq3, scales 5-9% inflated) than
//! the vq3 the rest of the model uses; running it under the glm52-fp8 router degenerates
//! all-int4 decode. This decodes each vq3 expert to f32 and re-quantizes with `quant_i4`
//! (R~0.98, self-consistent). Writes `L{l}.i4` in the same layout as `pack_i4`.
//!
//! usage: vq3_to_i4 <artifact-dir> [--layers N]   (back up existing L*.i4 first!)
use anyhow::{Context, Result, bail};
use rivoli::format::load_codebooks;
use rivoli::model::ModelConfig;
use rivoli::quant::{
    VQ_DIM, VQ_GROUP, VQ_INDEX_BITS, VqProj, i4_expert_stride, i4_row_bytes, i4_slot_offsets,
    quant_i4, vq_expert, vq_expert_bytes, vq_expert_stride, vq_groups, vq_row_bytes, VQ_ALIGN,
};
use std::fs::File;
use std::io::Write;
use std::os::unix::fs::FileExt;

fn get_idx(row: &[u8], k: usize) -> usize {
    let (base, shift) = (k * VQ_INDEX_BITS / 8, (k * VQ_INDEX_BITS) % 8);
    (((row[base] as u16 | (row[base + 1] as u16) << 8) >> shift) & 0xFFF) as usize
}

/// Decode one VQ projection to a dense f32 row-major `W[o_dim, i_dim]`.
fn decode_proj(p: &VqProj, cb: &[f32]) -> Vec<f32> {
    let (o_dim, i_dim) = (p.o_dim, p.i_dim);
    let (rb, ng, nsub) = (vq_row_bytes(i_dim), vq_groups(i_dim), i_dim / VQ_DIM);
    let mut w = vec![0f32; o_dim * i_dim];
    for o in 0..o_dim {
        let ir = &p.indices[o * rb..(o + 1) * rb];
        for k in 0..nsub {
            let g = (o * ng + (k * VQ_DIM) / VQ_GROUP) * 2;
            let s = rivoli::math::bf16_to_f32(u16::from_le_bytes([p.scales[g], p.scales[g + 1]]));
            let idx = get_idx(ir, k);
            let c = &cb[idx * VQ_DIM..idx * VQ_DIM + VQ_DIM];
            for d in 0..VQ_DIM {
                w[o * i_dim + k * VQ_DIM + d] = s * c[d];
            }
        }
    }
    w
}

/// Build one expert's `.i4` block (vq expert bytes -> int4, per `i4_slot_offsets`).
fn build_block(vqb: &[u8], cbs: &[Vec<f32>; 3], off: &[usize; 6], stride: usize, h: usize, m: usize) -> Vec<u8> {
    let projs = vq_expert(vqb, 0, h, m);
    let mut blk = vec![0u8; stride];
    for k in 0..3 {
        let w = decode_proj(&projs[k], &cbs[k]);
        let (packed, scale) = quant_i4(&w, projs[k].o_dim, projs[k].i_dim);
        let po = off[k * 2];
        blk[po..po + packed.len()].copy_from_slice(&packed);
        let so = off[k * 2 + 1];
        for (j, s) in scale.iter().enumerate() {
            blk[so + j * 4..so + j * 4 + 4].copy_from_slice(&s.to_le_bytes());
        }
        debug_assert_eq!(packed.len(), projs[k].o_dim * i4_row_bytes(projs[k].i_dim));
    }
    blk
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        bail!("usage: vq3_to_i4 <artifact-dir> [--layers N]");
    }
    let dir = &args[1];
    let cfg = ModelConfig::load(dir).context("load manifest")?;
    let (h, m, ne) = (cfg.hidden, cfg.moe_inter, cfg.n_experts);
    let vq_stride = vq_expert_stride(h, m);
    let i4_stride = i4_expert_stride(h, m);
    let off = i4_slot_offsets(h, m);
    let cbs = load_codebooks(dir)?;
    let last = cfg.dense_layers
        + args
            .iter()
            .position(|a| a == "--layers")
            .and_then(|i| args.get(i + 1))
            .and_then(|s| s.parse().ok())
            .unwrap_or(cfg.n_layers - cfg.dense_layers);
    let nthreads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
    eprintln!("vq3->i4: layers {}..{last}, {}+1 experts, {nthreads} threads", cfg.dense_layers, ne);

    for l in cfg.dense_layers..last {
        let vqf = File::open(format!("{dir}/L{l:02}.vq3")).with_context(|| format!("open L{l:02}.vq3"))?;
        let ebytes = vq_expert_bytes(h, m);
        // read all (ne+1) vq expert blocks for this layer
        let n = ne + 1;
        let mut vqbufs: Vec<Vec<u8>> = Vec::with_capacity(n);
        for e in 0..n {
            let mut b = vec![0u8; ebytes];
            vqf.read_exact_at(&mut b, (VQ_ALIGN + e * vq_stride) as u64)
                .with_context(|| format!("read L{l:02} expert {e}"))?;
            vqbufs.push(b);
        }
        // parallel decode+requant into indexed blocks
        let mut blocks: Vec<Vec<u8>> = vec![Vec::new(); n];
        std::thread::scope(|sc| {
            let chunks: Vec<&mut [Vec<u8>]> = blocks.chunks_mut(n.div_ceil(nthreads)).collect();
            let vqref = &vqbufs;
            let (cbref, offref) = (&cbs, &off);
            for (ci, chunk) in chunks.into_iter().enumerate() {
                let base = ci * n.div_ceil(nthreads);
                sc.spawn(move || {
                    for (j, slot) in chunk.iter_mut().enumerate() {
                        *slot = build_block(&vqref[base + j], cbref, offref, i4_stride, h, m);
                    }
                });
            }
        });
        let path = format!("{dir}/L{l:02}.i4");
        let mut out = File::create(&path).with_context(|| format!("create {path}"))?;
        for b in &blocks {
            out.write_all(b)?;
        }
        eprintln!("  L{l:02}.i4 rewritten from vq3 ({n} blocks)");
    }
    eprintln!("done");
    Ok(())
}
