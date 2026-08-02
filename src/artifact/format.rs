//! The on-disk artifact: the single source of truth for the layout the converter
//! writes and the loader reads. An artifact directory is:
//!
//! ```text
//! manifest.json          # HF config fields + a `format` section (FormatMeta)
//! codebooks.f32          # 3 per-projection codebooks (gate, up, down): VQ_K·VQ_DIM f32 each
//! resident.safetensors   # every resident weight (fp8 attn/dense, int8 embed, f32 norms, bf16 indexer)
//! L{03..NN}.vq3          # per MoE layer: header + (n_experts + 1) expert blocks,
//!                        #   block = gate‖up‖down; block n_experts = the shared expert
//! L{03..NN}.i4           # optional int4 twin of the .vq3 (bin/fp8_to_i4): headerless,
//!                        #   (n_experts + 1) blocks from offset 0. Streamed under --i4.
//! ```

use anyhow::{Context, Result, ensure};
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::os::fd::{AsRawFd, RawFd};

use crate::artifact::quant::{
    VQ_DIM, VQ_GROUP, VQ_INDEX_BITS, VQ_K, i4_expert_bytes, i4_expert_stride, vq_expert_bytes,
    vq_expert_stride,
};

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
        // The fp8 GEMV kernels index the block scale with a SHIFT (`blk_shift`), so a
        // non-power-of-two tile is unrunnable. The kernel launchers reject it too (arg
        // guard 1003); catching it here turns a mid-decode HIP error into a startup
        // message that names the offending value.
        ensure!(
            m.fp8_block > 0 && m.fp8_block.is_power_of_two(),
            "artifact fp8_block ({}) must be a power of two",
            m.fp8_block
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

    fn name(self) -> &'static str {
        match self {
            Dtype::F32 => "F32",
            Dtype::U8 => "U8",
            Dtype::I8 => "I8",
            Dtype::Bf16 => "BF16",
            Dtype::F8E4M3 => "F8_E4M3",
        }
    }
}

/// Minimal safetensors writer for the resident artifact — collects tensors, then
/// serializes `u64 header_len ‖ JSON header ‖ concatenated data`. Owns each tensor's
/// bytes until `write` (the resident set is ~10 GiB, held once in host RAM).
#[derive(Default)]
pub struct SafeWriter {
    tensors: Vec<(String, Dtype, Vec<usize>, Vec<u8>)>,
}

impl SafeWriter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(
        &mut self,
        name: impl Into<String>,
        dtype: Dtype,
        shape: Vec<usize>,
        bytes: Vec<u8>,
    ) {
        self.tensors.push((name.into(), dtype, shape, bytes));
    }

    /// Copy an fp8 tensor (`<name>.weight` F8E4M3 + `.weight_scale_inv` F32) from a
    /// reader verbatim — the resident attn/dense/indexer projections.
    pub fn copy_fp8(&mut self, src: &Safetensors, name: &str) -> Result<()> {
        let (w, shape) = src.typed(&format!("{name}.weight"), Dtype::F8E4M3)?;
        self.add(
            format!("{name}.weight"),
            Dtype::F8E4M3,
            shape.to_vec(),
            w.to_vec(),
        );
        let (sc, ssh) = src.typed(&format!("{name}.weight_scale_inv"), Dtype::F32)?;
        self.add(
            format!("{name}.weight_scale_inv"),
            Dtype::F32,
            ssh.to_vec(),
            sc.to_vec(),
        );
        Ok(())
    }

    /// Add a bf16 tensor from a reader, widened to f32 (norms, router gate,
    /// weights_proj, k_norm — everything the loader reads as f32).
    pub fn add_widened(&mut self, src: &Safetensors, name: &str) -> Result<()> {
        let (bytes, shape) = src.typed(name, Dtype::Bf16)?;
        let f32b: Vec<u8> = bytes
            .chunks_exact(2)
            .flat_map(|c| crate::math::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])).to_le_bytes())
            .collect();
        self.add(name, Dtype::F32, shape.to_vec(), f32b);
        Ok(())
    }

    pub fn write(&self, path: &str) -> Result<()> {
        use std::io::Write;
        let mut hdr = serde_json::Map::new();
        let mut offset = 0usize;
        for (name, dtype, shape, bytes) in &self.tensors {
            let begin = offset;
            offset += bytes.len();
            hdr.insert(
                name.clone(),
                serde_json::json!({ "dtype": dtype.name(), "shape": shape, "data_offsets": [begin, offset] }),
            );
        }
        let hjson = serde_json::to_vec(&serde_json::Value::Object(hdr))?;
        let file = std::fs::File::create(path).with_context(|| format!("create {path}"))?;
        let mut f = std::io::BufWriter::new(file);
        f.write_all(&(hjson.len() as u64).to_le_bytes())?;
        f.write_all(&hjson)?;
        for (_, _, _, bytes) in &self.tensors {
            f.write_all(bytes)?;
        }
        f.flush()?;
        Ok(())
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

    /// Dequantize a block-scaled fp8 projection (`<name>.weight` F8E4M3 +
    /// `<name>.weight_scale_inv` F32) to row-major f32 `[o_dim, i_dim]`. The one
    /// fp8-read used by both converters (`bin/convert` → `.vq3`, `bin/fp8_to_i4` →
    /// `.i4`), so the two cannot drift on the decode convention.
    ///
    /// Both shapes are checked here rather than trusted: `weight_scale_inv` is
    /// `[ceil(o/block), ceil(i/block)]` row-major, and a scale tensor of the wrong
    /// extent would otherwise mis-tile silently — a wrong-but-plausible dequant,
    /// which is far worse than a hard failure.
    pub fn dequant_fp8(
        &self,
        name: &str,
        o_dim: usize,
        i_dim: usize,
        block: usize,
    ) -> Result<Vec<f32>> {
        let (w, shape) = self.typed(&format!("{name}.weight"), Dtype::F8E4M3)?;
        ensure!(
            shape == [o_dim, i_dim],
            "{name}.weight: shape {shape:?} != [{o_dim},{i_dim}]"
        );
        let (sb, ssh) = self.typed(&format!("{name}.weight_scale_inv"), Dtype::F32)?;
        let want = [o_dim.div_ceil(block), i_dim.div_ceil(block)];
        ensure!(
            ssh == want,
            "{name}.weight_scale_inv: shape {ssh:?} != {want:?} (block {block})"
        );
        let scale = crate::artifact::quant::read_f32(sb);
        Ok(crate::artifact::quant::dequant_fp8_block(
            w, &scale, o_dim, i_dim, block,
        ))
    }
}

/// Provenance of the artifact's `.i4` expert set: which tool produced it, from what,
/// and over which layers. Absent on artifacts built before this field existed — and
/// that absence is itself the signal, since a `vq3_to_i4` set and an `fp8_to_i4` set are
/// otherwise byte-indistinguishable on disk, which is exactly how a bad `.i4` set
/// stayed invisible.
///
/// EVERY writer of `L{l}.i4` must call [`I4Source::stamp`]. A stale stamp is worse
/// than no stamp: the engine reports it as fact, so a tool that rewrites the set
/// without restamping turns an honest ambiguity into a confident lie.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct I4Source {
    /// Binary that wrote the set, e.g. `"fp8_to_i4"`.
    pub tool: String,
    /// Derivation chain, e.g. `"fp8->int4"` vs the older `"fp8->vq3->int4"`.
    pub chain: String,
    /// Path of the source the weights were derived from.
    pub src: String,
    /// Half-open layer range this provenance covers, `[from, to)`. A run that
    /// rebuilds only part of the set records only that part, so a mixed artifact
    /// never claims to be uniform.
    pub layers: [usize; 2],
    /// [`crate::artifact::quant::I4_GROUP`] the set was quantized at — weights per f32 scale
    /// along the input dim. Without it a G=128 set and a G=64 set are the same
    /// `tool`/`chain`/`src` triple, and the quality difference between them would be
    /// unattributable. `None` is an artifact predating group scales (per-row); such a
    /// set has a different `.i4` file size and is rejected by `ExpertSet::open`, so
    /// this is a diagnosis, not a load-time guard.
    #[serde(default)]
    pub group: Option<usize>,
}

impl I4Source {
    /// Read `<dir>/manifest.json`'s `i4_source` section. `Ok(None)` means "no
    /// manifest, or no such field" — an unstamped artifact, which is a reportable
    /// fact rather than an error. A field that is PRESENT but unparseable is an
    /// error: silently reporting it as "unstamped" would hide a real corruption.
    pub fn load(dir: &str) -> Result<Option<Self>> {
        let Ok(text) = std::fs::read(format!("{dir}/manifest.json")) else {
            return Ok(None);
        };
        let v: serde_json::Value =
            serde_json::from_slice(&text).with_context(|| format!("parse {dir}/manifest.json"))?;
        let Some(f) = v.get("i4_source") else {
            return Ok(None);
        };
        Ok(Some(
            serde_json::from_value(f.clone()).context("manifest i4_source is malformed")?,
        ))
    }

    /// Record this provenance in `<dir>/manifest.json`, merging with an existing
    /// stamp when the two describe the same derivation over adjoining layers — so a
    /// run resumed with `--from` still ends up claiming the whole set it rebuilt,
    /// rather than only its own final leg.
    ///
    /// Written tmp→fsync→rename: a torn `manifest.json` bricks an artifact whose
    /// `.i4` set alone is ~365 GB.
    pub fn stamp(&self, dir: &str) -> Result<()> {
        let path = format!("{dir}/manifest.json");
        let mut m: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).with_context(|| format!("read {path}"))?)
                .with_context(|| format!("parse {path}"))?;
        // An unreadable prior stamp is not an error HERE — we are replacing it. Only
        // the reader is strict, so a corrupt field can never be misreported as fact.
        let merged = match Self::load(dir).ok().flatten() {
            // Same derivation and the ranges touch → one contiguous claim. `group` is
            // part of "same derivation": merging a G=64 run into a G=128 run would
            // claim one uniform set for two incompatible formats.
            Some(p)
                if (p.tool.as_str(), p.chain.as_str(), p.src.as_str(), p.group)
                    == (&self.tool, &self.chain, &self.src, self.group)
                    && p.layers[0] <= self.layers[1]
                    && self.layers[0] <= p.layers[1] =>
            {
                Self {
                    layers: [
                        p.layers[0].min(self.layers[0]),
                        p.layers[1].max(self.layers[1]),
                    ],
                    ..self.clone()
                }
            }
            _ => self.clone(),
        };
        m["i4_source"] = serde_json::to_value(&merged)?;
        let tmp = format!("{path}.tmp");
        let mut f = std::fs::File::create(&tmp).with_context(|| format!("create {tmp}"))?;
        {
            use std::io::Write;
            f.write_all(&serde_json::to_vec_pretty(&m)?)?;
        }
        f.sync_all().with_context(|| format!("fsync {tmp}"))?;
        drop(f);
        std::fs::rename(&tmp, &path).with_context(|| format!("rename {tmp} -> {path}"))?;
        Ok(())
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

/// The per-layer expert files, opened O_DIRECT — one type for BOTH routed formats
/// (`.vq3` and `.i4`), which differ only by the block-start offset `hbytes` (one
/// aligned header block for `.vq3`, 0 for the headerless `.i4`) and `.vq3`'s header
/// validation on open. Routed experts (0..n_experts) stream via [`read_spec`]; the
/// shared expert (block n_experts) is read once for resident placement via
/// [`shared_block`]. An `.i4` block is gate‖gate_scale‖up‖up_scale‖down‖down_scale.
/// The dimensions an expert set is opened against. `n_layers` is EXCLUSIVE and is the
/// CALLER's bound, not the artifact's — the pin opens one layer past `cfg.n_layers` when
/// the MTP head has an expert file.
///
/// A struct rather than five positional `usize`s, because nothing about a transposed
/// `(hidden, moe_inter)` fails a check: the set opens, every length matches, and it streams
/// the wrong bytes. Five bare `usize`s in a row is the argument list that gets transposed.
#[derive(Clone, Copy)]
pub struct SetDims {
    pub dense_layers: usize,
    pub n_layers: usize,
    pub n_experts: usize,
    pub hidden: usize,
    pub moe_inter: usize,
}

pub struct ExpertSet {
    files: Vec<std::fs::File>, // O_DIRECT, index = layer - dense_layers
    dense_layers: usize,
    n_layers: usize,
    n_experts: usize,
    stride: usize,
    expert_bytes: usize,
    hbytes: usize, // block 0 starts at hbytes (aligned): VQ_ALIGN for .vq3, 0 for .i4
}

impl ExpertSet {
    /// Open the routed expert set in one format. `.vq3` reserves one aligned block for a
    /// header and validates it against these dims; `.i4` is headerless and starts at 0.
    ///
    /// ONE opener rather than two, because those three things (extension, block-start
    /// offset, header validation) are the entire difference and the pin chooses between
    /// them at RUNTIME — two entry points meant it wrote the identical six-argument
    /// dimension list twice, and nothing about a transposed `(hidden, moe_inter)` fails a
    /// length check: both sets open and stream the wrong bytes.
    pub fn open_routed(dir: &str, i4: bool, d: SetDims) -> Result<Self> {
        let (hidden, moe_inter) = (d.hidden, d.moe_inter);
        let (ext, hbytes, stride, expert_bytes) = if i4 {
            (
                "i4",
                0,
                i4_expert_stride(hidden, moe_inter),
                i4_expert_bytes(hidden, moe_inter),
            )
        } else {
            // One aligned block reserved for the header.
            let hb = crate::artifact::quant::VQ_ALIGN;
            (
                "vq3",
                hb,
                vq_expert_stride(hidden, moe_inter),
                vq_expert_bytes(hidden, moe_inter),
            )
        };
        Self::open(dir, ext, hbytes, stride, expert_bytes, d, |path, l| {
            if i4 {
                return Ok(()); // headerless
            }
            // Validate the header via a separate buffered read (the O_DIRECT fd is
            // for the streamer). Dims must match the config.
            let hdr = std::fs::read(path)
                .ok()
                .filter(|b| b.len() >= VQ3_HEADER_BYTES)
                .with_context(|| format!("read {path} header"))?;
            let h = Vq3Header::from_bytes(&hdr)?;
            ensure!(
                h.layer as usize == l
                    && h.n_experts as usize == d.n_experts
                    && h.hidden as usize == hidden
                    && h.moe_inter as usize == moe_inter,
                "{path}: header dims disagree with config"
            );
            Ok(())
        })
    }

    /// The int3-VQ set by loose dimensions — `tests/artifact.rs` opens the shipped `.vq3`
    /// this way. The engine's format is a runtime choice, so it goes through
    /// [`ExpertSet::open_routed`] with a [`SetDims`] it names its fields into.
    pub fn open_vq3(
        dir: &str,
        dense_layers: usize,
        n_layers: usize,
        n_experts: usize,
        hidden: usize,
        moe_inter: usize,
    ) -> Result<Self> {
        let d = SetDims {
            dense_layers,
            n_layers,
            n_experts,
            hidden,
            moe_inter,
        };
        Self::open_routed(dir, false, d)
    }

    fn open(
        dir: &str,
        ext: &str,
        hbytes: usize,
        stride: usize,
        expert_bytes: usize,
        d: SetDims,
        validate: impl Fn(&str, usize) -> Result<()>,
    ) -> Result<Self> {
        let SetDims {
            dense_layers,
            n_layers,
            n_experts,
            ..
        } = d;
        let want = hbytes + (n_experts + 1) * stride; // header + routed + shared
        let mut files = Vec::with_capacity(n_layers - dense_layers);
        for l in dense_layers..n_layers {
            let path = format!("{dir}/L{l:02}.{ext}");
            let f = open_direct(&path).with_context(|| format!("open {path}"))?;
            let len = f.metadata()?.len() as usize;
            ensure!(len == want, "{path}: {len} bytes, expected {want}");
            validate(&path, l)?;
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

    /// Cold-read spec for a routed expert: `(fd, begin, useful_len)`, `begin` aligned.
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

    /// Read the shared expert's block (index `n_experts`) for resident placement, via a
    /// one-time buffered aligned pread.
    pub fn shared_block(&self, layer: usize) -> Result<Vec<u8>> {
        use std::os::unix::fs::FileExt;
        ensure!(
            (self.dense_layers..self.n_layers).contains(&layer),
            "layer {layer} out of MoE range"
        );
        let off = self.hbytes + self.n_experts * self.stride;
        // O_DIRECT needs aligned buffer+offset+len; read the full aligned stride.
        let f = &self.files[layer - self.dense_layers];
        let mut aligned = vec![0u8; self.stride];
        f.read_exact_at(&mut aligned, off as u64)
            .with_context(|| format!("read shared block layer {layer}"))?;
        aligned.truncate(self.expert_bytes);
        Ok(aligned)
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
    let raw = crate::artifact::quant::read_f32(&std::fs::read(format!("{dir}/codebooks.f32"))?);
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

#[cfg(test)]
#[allow(clippy::unwrap_used)] // test setup: panic-on-failure is the readable idiom
mod tests {
    use super::*;

    #[test]
    fn vq3_header_roundtrips() {
        let h = Vq3Header::new(7, 256, 6144, 2048);
        let back = Vq3Header::from_bytes(&h.to_bytes()).unwrap();
        assert_eq!(back.magic, VQ3_MAGIC);
        assert_eq!(back.layer, 7);
        assert_eq!(back.n_experts, 256);
        assert_eq!(back.hidden, 6144);
        assert_eq!(back.moe_inter, 2048);
        assert_eq!(back.stride, vq_expert_stride(6144, 2048) as u64);
        // a corrupt magic must fail
        let mut bad = h.to_bytes();
        bad[0] = b'X';
        assert!(Vq3Header::from_bytes(&bad).is_err());
    }

    #[test]
    fn safewriter_roundtrips_through_reader() {
        let dir = format!(
            "{}/fmt_test",
            std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".into())
        );
        std::fs::create_dir_all(&dir).unwrap();
        let path = format!("{dir}/r.safetensors");
        let mut w = SafeWriter::new();
        let a: Vec<u8> = (0..48u8).collect(); // F32 [2,6]
        let b: Vec<u8> = (100..108u8).collect(); // I8 [2,4]
        w.add("a", Dtype::F32, vec![2, 6], a.clone());
        w.add("b", Dtype::I8, vec![2, 4], b.clone());
        w.write(&path).unwrap();

        let st = Safetensors::open_file(&path).unwrap();
        assert_eq!(st.shape("a").unwrap(), &[2, 6]);
        let (bb, sh) = st.typed("b", Dtype::I8).unwrap();
        assert_eq!(bb, &b[..]);
        assert_eq!(sh, &[2, 4]);
        assert!(st.typed("a", Dtype::I8).is_err()); // dtype mismatch fails loud
        assert!(!st.has("missing"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `dequant_fp8` must apply the right block scale to the right tile, and must
    /// REJECT a `weight_scale_inv` of the wrong extent — a silently mis-tiled scale
    /// grid is the failure this guard exists to prevent, and it produces plausible
    /// weights rather than an obvious error.
    #[test]
    fn dequant_fp8_tiles_scales_and_rejects_bad_shapes() {
        let dir = std::env::temp_dir().join("rivoli_deqfp8_test");
        let dir = dir.to_string_lossy().to_string();
        std::fs::create_dir_all(&dir).unwrap();
        let path = format!("{dir}/w.safetensors");
        // A [2,4] fp8 matrix with block=2 → a [1,2] scale grid: the left 2 columns
        // scale by 10, the right 2 by 100. A row/column-swapped tiling gives
        // different answers here, so this pins the orientation.
        let mut w = SafeWriter::new();
        w.add(
            "t.weight",
            Dtype::F8E4M3,
            vec![2, 4],
            vec![0x38, 0x40, 0x38, 0xB8, 0xB8, 0x00, 0x40, 0x38], // 1,2,1,-1 / -1,0,2,1
        );
        w.add(
            "t.weight_scale_inv",
            Dtype::F32,
            vec![1, 2],
            [10.0f32, 100.0]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
        );
        w.write(&path).unwrap();
        let st = Safetensors::open_file(&path).unwrap();

        let got = st.dequant_fp8("t", 2, 4, 2).unwrap();
        assert_eq!(
            got,
            vec![10.0, 20.0, 100.0, -100.0, -10.0, 0.0, 200.0, 100.0]
        );

        // Wrong declared dims, and a scale grid of the wrong extent, both fail loud.
        assert!(st.dequant_fp8("t", 4, 2, 2).is_err());
        assert!(st.dequant_fp8("t", 2, 4, 1).is_err()); // would want a [2,4] grid
    }

    /// Provenance round-trips, a missing field reads as unstamped, a malformed one is
    /// an error (not silently "unstamped"), and a resumed run merges into one range.
    #[test]
    fn i4_source_round_trips_and_merges_adjoining_runs() {
        let dir = std::env::temp_dir().join("rivoli_i4src_test");
        let dir = dir.to_string_lossy().to_string();
        std::fs::create_dir_all(&dir).unwrap();
        let mf = format!("{dir}/manifest.json");
        std::fs::write(&mf, br#"{"hidden_size":6144}"#).unwrap();
        assert!(I4Source::load(&dir).unwrap().is_none()); // no field yet

        let a = I4Source {
            tool: "fp8_to_i4".into(),
            chain: "fp8->int4".into(),
            src: "/src".into(),
            layers: [3, 40],
            group: Some(crate::artifact::quant::I4_GROUP),
        };
        a.stamp(&dir).unwrap();
        assert_eq!(I4Source::load(&dir).unwrap().as_ref(), Some(&a));
        // The stamp must not clobber the rest of the manifest.
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&mf).unwrap()).unwrap();
        assert_eq!(v["hidden_size"], 6144);

        // A resume picking up where the first run stopped claims the whole range.
        I4Source {
            layers: [40, 78],
            ..a.clone()
        }
        .stamp(&dir)
        .unwrap();
        assert_eq!(I4Source::load(&dir).unwrap().unwrap().layers, [3, 78]);

        // A different derivation replaces rather than merges — no stale claims.
        let b = I4Source {
            chain: "fp8->vq3->int4".into(),
            layers: [3, 78],
            ..a.clone()
        };
        b.stamp(&dir).unwrap();
        assert_eq!(I4Source::load(&dir).unwrap(), Some(b.clone()));

        // A different GROUP is a different derivation too: rebuilding the tail at
        // G=64 must not merge into a G=128 claim over the head, or the manifest would
        // describe one uniform set where two incompatible formats sit side by side.
        let c = I4Source {
            group: Some(64),
            layers: [40, 78],
            ..b.clone()
        };
        c.stamp(&dir).unwrap();
        assert_eq!(I4Source::load(&dir).unwrap(), Some(c));

        // An artifact stamped before group scales existed reads as group: None.
        std::fs::write(
            &mf,
            br#"{"i4_source":{"tool":"t","chain":"c","src":"/s","layers":[0,1]}}"#,
        )
        .unwrap();
        assert_eq!(I4Source::load(&dir).unwrap().unwrap().group, None);

        // Present-but-malformed is an error, never a silent "unstamped".
        std::fs::write(&mf, br#"{"i4_source":{"tool":"x"}}"#).unwrap();
        assert!(I4Source::load(&dir).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
