//! The on-disk artifact: the single source of truth for the layout the converter
//! writes and the loader reads. An artifact directory is:
//!
//! ```text
//! manifest.json          # HF config fields + a `format` section (FormatMeta)
//! codebooks.f32          # 3 per-projection codebooks (gate, up, down): VQ_K·VQ_DIM f32 each
//! resident.safetensors   # every resident weight (fp8 attn/dense, int8 embed, f32 norms, bf16 indexer)
//! L{03..NN}.vq3          # per MoE layer: header + (n_experts + 1) expert blocks,
//!                        #   block = gate‖up‖down; block n_experts = the shared expert
//! ```

use anyhow::{Context, Result, ensure};
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::os::fd::{AsRawFd, RawFd};

use crate::quant::{VQ_DIM, VQ_GROUP, VQ_INDEX_BITS, VQ_K, vq_expert_bytes, vq_expert_stride};

/// The `format` section of `manifest.json` — everything the loader needs beyond
/// the HF config fields (which `ModelConfig` reads from the same file).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormatMeta {
    pub version: u32,
    pub vq_dim: usize,
    pub vq_k: usize,
    pub vq_index_bits: usize,
    pub vq_group: usize,
    /// fp8 `weight_scale_inv` tile size (128 for the GLM-5.2 checkpoint).
    pub fp8_block: usize,
}

impl FormatMeta {
    pub const VERSION: u32 = 1;

    /// The current build's parameters — what the converter stamps into the manifest.
    pub fn current(fp8_block: usize) -> Self {
        Self {
            version: Self::VERSION,
            vq_dim: VQ_DIM,
            vq_k: VQ_K,
            vq_index_bits: VQ_INDEX_BITS,
            vq_group: VQ_GROUP,
            fp8_block,
        }
    }

    /// Read `<dir>/manifest.json`'s `format` section and check it matches this build
    /// (VQ params are compiled into the kernels, so a mismatch is unrunnable).
    pub fn load(dir: &str) -> Result<Self> {
        let v: serde_json::Value =
            serde_json::from_slice(&std::fs::read(format!("{dir}/manifest.json"))?)
                .with_context(|| format!("parse {dir}/manifest.json"))?;
        let m: FormatMeta = serde_json::from_value(v["format"].clone())
            .context("manifest.json missing/invalid `format` section")?;
        ensure!(
            m.version == Self::VERSION,
            "artifact format v{} != build v{}",
            m.version,
            Self::VERSION
        );
        ensure!(
            m.vq_dim == VQ_DIM
                && m.vq_k == VQ_K
                && m.vq_index_bits == VQ_INDEX_BITS
                && m.vq_group == VQ_GROUP,
            "artifact VQ params differ from the compiled-in kernel params"
        );
        Ok(m)
    }
}

// ── safetensors reader (fp8 source shards + the resident artifact) ──────────────

/// A tensor's dtype in a safetensors header (only the ones this engine uses).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F32,
    U8,
    I8,
    Bf16,
    F8E4M3,
}

impl Dtype {
    fn parse(s: &str) -> Option<Dtype> {
        Some(match s {
            "F32" => Dtype::F32,
            "U8" => Dtype::U8,
            "I8" => Dtype::I8,
            "BF16" => Dtype::Bf16,
            "F8_E4M3" => Dtype::F8E4M3,
            _ => return None,
        })
    }
}

struct Loc {
    shard: usize,
    begin: usize,
    len: usize,
    dtype: Dtype,
    shape: Vec<usize>,
}

/// Read-only mmap of one or more safetensors files, tensors indexed by name. Serves
/// the multi-shard fp8 source (converter) and the single resident file (loader).
pub struct Safetensors {
    mmaps: Vec<Mmap>,
    index: HashMap<String, Loc>,
}

impl Safetensors {
    /// mmap + index every `*.safetensors` under `dir` (sorted).
    pub fn open_dir(dir: &str) -> Result<Self> {
        let mut paths: Vec<_> = std::fs::read_dir(dir)
            .with_context(|| format!("read dir {dir}"))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        paths.sort();
        ensure!(!paths.is_empty(), "no *.safetensors in {dir}");
        Self::open_files(&paths)
    }

    /// mmap + index one file.
    pub fn open_file(path: &str) -> Result<Self> {
        Self::open_files(std::slice::from_ref(&std::path::PathBuf::from(path)))
    }

    fn open_files(paths: &[std::path::PathBuf]) -> Result<Self> {
        let mut mmaps = Vec::with_capacity(paths.len());
        let mut index = HashMap::new();
        for path in paths {
            let file = std::fs::File::open(path).with_context(|| format!("open {path:?}"))?;
            // SAFETY: read-only for the reader's lifetime.
            let mmap = unsafe { Mmap::map(&file) }.with_context(|| format!("mmap {path:?}"))?;
            let hlen = u64::from_le_bytes(mmap[..8].try_into()?) as usize;
            let hdr: serde_json::Value = serde_json::from_slice(&mmap[8..8 + hlen])
                .with_context(|| format!("parse header {path:?}"))?;
            let base = 8 + hlen;
            let shard = mmaps.len();
            for (name, t) in hdr
                .as_object()
                .context("safetensors header not an object")?
            {
                if name == "__metadata__" {
                    continue;
                }
                let u = |v: &serde_json::Value| v.as_u64().context("non-integer header field");
                let off = t["data_offsets"].as_array().context("data_offsets")?;
                let (b, e) = (u(&off[0])? as usize, u(&off[1])? as usize);
                let dtype = Dtype::parse(t["dtype"].as_str().context("dtype")?)
                    .with_context(|| format!("unsupported dtype for {name}"))?;
                let shape = t["shape"]
                    .as_array()
                    .context("shape")?
                    .iter()
                    .map(|v| u(v).map(|x| x as usize))
                    .collect::<Result<_>>()?;
                index.insert(
                    name.clone(),
                    Loc {
                        shard,
                        begin: base + b,
                        len: e - b,
                        dtype,
                        shape,
                    },
                );
            }
            mmaps.push(mmap);
        }
        Ok(Self { mmaps, index })
    }

    fn loc(&self, name: &str) -> Result<&Loc> {
        self.index
            .get(name)
            .with_context(|| format!("tensor {name} not found"))
    }

    /// Raw bytes of a tensor.
    pub fn bytes(&self, name: &str) -> Result<&[u8]> {
        let l = self.loc(name)?;
        Ok(&self.mmaps[l.shard][l.begin..l.begin + l.len])
    }

    /// Bytes + dtype-check + shape of a tensor (fails loud on a dtype mismatch).
    pub fn typed(&self, name: &str, want: Dtype) -> Result<(&[u8], &[usize])> {
        let l = self.loc(name)?;
        ensure!(
            l.dtype == want,
            "{name}: dtype {:?}, expected {want:?}",
            l.dtype
        );
        Ok((&self.mmaps[l.shard][l.begin..l.begin + l.len], &l.shape))
    }

    pub fn shape(&self, name: &str) -> Result<&[usize]> {
        Ok(&self.loc(name)?.shape)
    }

    pub fn has(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }
}

// ── .vq3 expert files (streamed routed experts + resident shared) ───────────────

/// `.vq3` per-file header. Little-endian, 40 bytes, at the start of every layer file.
/// Self-describing: a dim/version mismatch or truncation fails loud on open.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Vq3Header {
    pub magic: [u8; 4], // b"VQ3\0"
    pub version: u32,
    pub layer: u32,
    pub n_experts: u32, // ROUTED experts; the file holds n_experts + 1 blocks (last = shared)
    pub hidden: u32,
    pub moe_inter: u32,
    pub stride: u64, // per-expert O_DIRECT-aligned block stride
    pub reserved: u64,
}

pub const VQ3_MAGIC: [u8; 4] = *b"VQ3\0";
pub const VQ3_HEADER_BYTES: usize = 40;

impl Vq3Header {
    pub fn new(layer: usize, n_experts: usize, hidden: usize, moe_inter: usize) -> Self {
        Self {
            magic: VQ3_MAGIC,
            version: FormatMeta::VERSION,
            layer: layer as u32,
            n_experts: n_experts as u32,
            hidden: hidden as u32,
            moe_inter: moe_inter as u32,
            stride: vq_expert_stride(hidden, moe_inter) as u64,
            reserved: 0,
        }
    }

    /// Serialize to the 40-byte on-disk header.
    pub fn to_bytes(&self) -> [u8; VQ3_HEADER_BYTES] {
        let mut b = [0u8; VQ3_HEADER_BYTES];
        b[0..4].copy_from_slice(&self.magic);
        b[4..8].copy_from_slice(&self.version.to_le_bytes());
        b[8..12].copy_from_slice(&self.layer.to_le_bytes());
        b[12..16].copy_from_slice(&self.n_experts.to_le_bytes());
        b[16..20].copy_from_slice(&self.hidden.to_le_bytes());
        b[20..24].copy_from_slice(&self.moe_inter.to_le_bytes());
        b[24..32].copy_from_slice(&self.stride.to_le_bytes());
        b[32..40].copy_from_slice(&self.reserved.to_le_bytes());
        b
    }

    fn from_bytes(b: &[u8]) -> Result<Self> {
        ensure!(b.len() >= VQ3_HEADER_BYTES, "vq3 header short");
        let h = Self {
            magic: b[0..4].try_into()?,
            version: u32::from_le_bytes(b[4..8].try_into()?),
            layer: u32::from_le_bytes(b[8..12].try_into()?),
            n_experts: u32::from_le_bytes(b[12..16].try_into()?),
            hidden: u32::from_le_bytes(b[16..20].try_into()?),
            moe_inter: u32::from_le_bytes(b[20..24].try_into()?),
            stride: u64::from_le_bytes(b[24..32].try_into()?),
            reserved: u64::from_le_bytes(b[32..40].try_into()?),
        };
        ensure!(h.magic == VQ3_MAGIC, "bad .vq3 magic");
        ensure!(
            h.version == FormatMeta::VERSION,
            ".vq3 version {} != {}",
            h.version,
            FormatMeta::VERSION
        );
        Ok(h)
    }
}

/// The per-layer expert files, opened O_DIRECT. Routed experts (0..n_experts) stream
/// via [`read_spec`]; the shared expert (block n_experts) is read once for resident
/// placement via [`shared_block`]. Header-validated against `cfg` on open.
pub struct Vq3Set {
    files: Vec<std::fs::File>, // O_DIRECT, index = layer - dense_layers
    dense_layers: usize,
    n_layers: usize,
    n_experts: usize,
    stride: usize,
    expert_bytes: usize,
    hbytes: usize, // header bytes, block 0 starts at hbytes (aligned)
}

impl Vq3Set {
    /// Open every `<dir>/L{ll}.vq3`, validating each header against the config dims.
    /// Block 0 begins at a `VQ_ALIGN`-aligned offset after the header so routed reads
    /// stay O_DIRECT-aligned.
    pub fn open(
        dir: &str,
        dense_layers: usize,
        n_layers: usize,
        n_experts: usize,
        hidden: usize,
        moe_inter: usize,
    ) -> Result<Self> {
        let stride = vq_expert_stride(hidden, moe_inter);
        let expert_bytes = vq_expert_bytes(hidden, moe_inter);
        let hbytes = crate::quant::VQ_ALIGN; // one aligned block reserved for the header
        let blocks = n_experts + 1; // routed + shared
        let want = hbytes + blocks * stride;
        let mut files = Vec::with_capacity(n_layers - dense_layers);
        for l in dense_layers..n_layers {
            let path = format!("{dir}/L{l:02}.vq3");
            let f = open_direct(&path).with_context(|| format!("open {path}"))?;
            let len = f.metadata()?.len() as usize;
            ensure!(len == want, "{path}: {len} bytes, expected {want}");
            // Validate the header (a small buffered read; the O_DIRECT fd is only for
            // the streamer, so read the header via a separate buffered handle).
            let hdr = std::fs::read(&path)
                .ok()
                .filter(|b| b.len() >= VQ3_HEADER_BYTES);
            let hdr = hdr.with_context(|| format!("read {path} header"))?;
            let h = Vq3Header::from_bytes(&hdr)?;
            ensure!(
                h.layer as usize == l
                    && h.n_experts as usize == n_experts
                    && h.hidden as usize == hidden
                    && h.moe_inter as usize == moe_inter,
                "{path}: header dims disagree with config"
            );
            files.push(f);
        }
        Ok(Self {
            files,
            dense_layers,
            n_layers,
            n_experts,
            stride,
            expert_bytes,
            hbytes,
        })
    }

    /// Cold-read spec for a routed expert: `(fd, begin, useful_len)`, `begin` VQ_ALIGN-aligned.
    pub fn read_spec(&self, layer: usize, expert: usize) -> Result<(RawFd, usize, usize)> {
        ensure!(
            (self.dense_layers..self.n_layers).contains(&layer),
            "layer {layer} out of MoE range"
        );
        ensure!(
            expert < self.n_experts,
            "expert {expert} >= {}",
            self.n_experts
        );
        let fd = self.files[layer - self.dense_layers].as_raw_fd();
        Ok((fd, self.hbytes + expert * self.stride, self.expert_bytes))
    }

    /// Read the shared expert's block (index `n_experts`) for resident placement.
    /// Uses a buffered pread (loaded once at startup).
    pub fn shared_block(&self, layer: usize) -> Result<Vec<u8>> {
        use std::os::unix::fs::FileExt;
        ensure!(
            (self.dense_layers..self.n_layers).contains(&layer),
            "layer {layer} out of MoE range"
        );
        let off = self.hbytes + self.n_experts * self.stride;
        // O_DIRECT fd can't do an unaligned buffered read; reopen buffered for this
        // one-time startup read. (Path reconstructed from the layer.)
        let mut buf = vec![0u8; self.expert_bytes];
        let f = &self.files[layer - self.dense_layers];
        // The O_DIRECT fd requires aligned buffer+offset+len; `off` and expert_bytes'
        // superset are aligned via the block stride, so read the full aligned stride.
        let mut aligned = vec![0u8; self.stride];
        f.read_exact_at(&mut aligned, off as u64)
            .with_context(|| format!("read shared block layer {layer}"))?;
        buf.copy_from_slice(&aligned[..self.expert_bytes]);
        Ok(buf)
    }

    pub fn expert_slot(&self) -> usize {
        self.stride
    }
}

/// Open a file O_DIRECT (page-cache-bypassing NVMe DMA).
fn open_direct(path: &str) -> Result<std::fs::File> {
    use std::os::unix::fs::OpenOptionsExt;
    Ok(std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(path)?)
}

/// Load the 3 per-projection codebooks from `<dir>/codebooks.f32` (gate, up, down),
/// each `VQ_K·VQ_DIM` f32, concatenated in that order.
pub fn load_codebooks(dir: &str) -> Result<[Vec<f32>; 3]> {
    let raw = crate::quant::read_f32(&std::fs::read(format!("{dir}/codebooks.f32"))?);
    ensure!(
        raw.len() == 3 * VQ_K * VQ_DIM,
        "codebooks.f32: {} f32, expected {}",
        raw.len(),
        3 * VQ_K * VQ_DIM
    );
    let n = VQ_K * VQ_DIM;
    Ok([
        raw[..n].to_vec(),
        raw[n..2 * n].to_vec(),
        raw[2 * n..].to_vec(),
    ])
}
