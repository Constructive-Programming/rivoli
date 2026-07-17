//! Snapshot reader: mmap every `out-*.safetensors` shard, parse its header,
//! and build one name → location index. The index is the map the feed side
//! streams against; expert weights are NOT copied here — only located.
//!
//! safetensors layout per shard: `[u64 LE header_len][JSON header][raw data]`.
//! The JSON maps tensor name → {dtype, shape, data_offsets:[begin,end]} where
//! offsets are relative to the start of the data section (= 8 + header_len).
//!
//! int4 experts are stored as two tensors: `<name>.weight` (per-row packed
//! nibbles, `(lo+8)|((hi+8)<<4)`) and `<name>.weight.qs` (F32 per-row scale).
//! Dequant lands in M2 against colibri's kernel as the oracle; M0 only indexes.

use anyhow::{Context, Result, bail};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

/// On-disk element type. The snapshot uses exactly two: `F32` (norm weights,
/// the router gate, and per-row scales `.qs`) and `U8` (packed int4 expert
/// nibbles, and the int8 embedding/lm_head). Any other dtype is rejected at
/// index time. Expert weights are int4-only; the int8 U8 tensors are a small
/// distinct class reached through their own accessors, never [`Int4Matrix`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F32,
    U8,
}

impl Dtype {
    fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "F32" => Dtype::F32,
            "U8" => Dtype::U8,
            other => bail!("unsupported tensor dtype {other:?} (this engine is int4-only)"),
        })
    }
}

/// A per-row int4 weight matrix located in the mmap: `packed` nibbles and their
/// per-output-row `scale` bytes, borrowed zero-copy. The only way to obtain one
/// is [`Snapshot::int4`], which validates the `.weight`/`.qs` pairing and the
/// byte lengths — so "an expert weight that isn't int4" is unrepresentable.
#[derive(Debug, Clone, Copy)]
pub struct Int4Matrix<'a> {
    /// `o_dim` rows × `row_bytes(i_dim)` packed nibbles (`(nibble-8)*scale`).
    pub packed: &'a [u8],
    /// `o_dim` little-endian f32 scales (raw bytes; decoded per row on read).
    pub scale: &'a [u8],
    pub o_dim: usize,
    pub i_dim: usize,
}

/// Where one tensor's bytes live: which mmap'd shard and the byte range within.
#[derive(Debug, Clone)]
pub struct TensorLoc {
    pub shard: usize,
    pub begin: usize,
    pub end: usize,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
}

impl TensorLoc {
    pub fn nbytes(&self) -> usize {
        self.end - self.begin
    }
}

/// One raw header entry as written by safetensors.
#[derive(serde::Deserialize)]
struct RawTensor {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

/// One shard's mmap plus the tensors located within it (offsets absolute).
struct IndexedShard {
    mmap: Mmap,
    entries: Vec<(String, TensorLoc)>,
}

pub struct Snapshot {
    shards: Vec<Mmap>,
    index: HashMap<String, TensorLoc>,
}

impl Snapshot {
    /// mmap and index every shard under `dir`. Fails if no shards are found.
    pub fn open(dir: &str) -> Result<Self> {
        let mut paths: Vec<_> = std::fs::read_dir(dir)
            .with_context(|| format!("read snapshot dir {dir}"))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("out-") && n.ends_with(".safetensors"))
            })
            .collect();
        paths.sort();
        if paths.is_empty() {
            bail!("no out-*.safetensors shards in {dir}");
        }

        let mut shards = Vec::with_capacity(paths.len());
        let mut index = HashMap::new();

        for (si, path) in paths.iter().enumerate() {
            let shard = Self::index_shard(path)
                .with_context(|| format!("index shard {}", path.display()))?;
            for (name, loc) in shard.entries {
                // A name appearing in two shards means a corrupt/misassembled
                // snapshot — fail loudly rather than silently keep the last.
                if let Some(prev) = index.insert(name.clone(), TensorLoc { shard: si, ..loc }) {
                    bail!("tensor {name} present in shard {} and {si}", prev.shard);
                }
            }
            shards.push(shard.mmap);
        }
        Ok(Self { shards, index })
    }

    fn index_shard(path: &Path) -> Result<IndexedShard> {
        let file = File::open(path).context("open shard")?;
        // SAFETY: the shard is a read-only model file; we never mutate the map
        // and it outlives no borrow past Snapshot's own lifetime.
        let mmap = unsafe { Mmap::map(&file) }.context("mmap shard")?;
        if mmap.len() < 8 {
            bail!("shard shorter than 8-byte header length");
        }
        let hlen = u64::from_le_bytes(mmap[0..8].try_into()?) as usize;
        // Reject an absurd declared header before allocating against it.
        if hlen > (512 << 20) || 8 + hlen > mmap.len() {
            bail!("implausible safetensors header length {hlen}");
        }
        let data_start = 8 + hlen;
        let data_len = mmap.len() - data_start;

        // The header is a flat object of {name: RawTensor}, plus an optional
        // "__metadata__" object we skip via serde's untagged tolerance below.
        let raw: HashMap<String, serde_json::Value> = serde_json::from_slice(&mmap[8..data_start])
            .context("parse safetensors header json")?;

        let mut entries = Vec::with_capacity(raw.len());
        for (name, val) in raw {
            if name == "__metadata__" {
                continue;
            }
            let t: RawTensor = serde_json::from_value(val)
                .with_context(|| format!("tensor {name} header fields"))?;
            let [begin, end] = t.data_offsets;
            // Validate offsets before trusting them as slice bounds.
            if begin > end || end > data_len {
                bail!("tensor {name} offsets [{begin},{end}] out of data range {data_len}");
            }
            entries.push((
                name,
                TensorLoc {
                    shard: 0, // set by caller
                    begin: data_start + begin,
                    end: data_start + end,
                    dtype: Dtype::parse(&t.dtype)?,
                    shape: t.shape,
                },
            ));
        }
        Ok(IndexedShard { mmap, entries })
    }

    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    pub fn get(&self, name: &str) -> Option<&TensorLoc> {
        self.index.get(name)
    }

    /// Bytes of a required tensor, failing loudly with the name if missing —
    /// the decode path wants context, not a silent `None` at every call site.
    pub fn require(&self, name: &str) -> Result<&[u8]> {
        self.bytes(name)
            .with_context(|| format!("required tensor {name} not found in snapshot"))
    }

    /// Locate an int4 weight matrix `W[o_dim, i_dim]` by base name (expects
    /// `<name>.weight` U8 packed nibbles + `<name>.weight.qs` F32 per-row
    /// scales). Weight/scale tensors are stored 1-D (raw byte/element counts,
    /// not `[o,i]`), so `i_dim` — known from the model config (hidden or
    /// moe_inter) — is supplied; `o_dim` is derived from the scale count and
    /// cross-checked against the packed byte count, which also rejects a tensor
    /// that isn't int4-packed (e.g. an int8 embedding). Only constructor of
    /// [`Int4Matrix`], so every downstream expert read is a valid int4 matrix.
    pub fn int4(&self, name: &str, i_dim: usize) -> Result<Int4Matrix<'_>> {
        let wname = format!("{name}.weight");
        let sname = format!("{name}.weight.qs");
        let wdt = self
            .index
            .get(&wname)
            .with_context(|| format!("int4 weight {wname} not found"))?
            .dtype;
        let sdt = self
            .index
            .get(&sname)
            .with_context(|| format!("int4 scale {sname} not found"))?
            .dtype;
        if wdt != Dtype::U8 {
            bail!("{wname} is {wdt:?}, expected U8 packed int4");
        }
        if sdt != Dtype::F32 {
            bail!("{sname} is {sdt:?}, expected F32 scale");
        }
        let packed = self.require(&wname)?;
        let scale = self.require(&sname)?;
        let o_dim = scale.len() / 4;
        let want_packed = o_dim * crate::quant::row_bytes(i_dim);
        if packed.len() != want_packed {
            bail!(
                "{wname}: {} packed bytes for o_dim={o_dim} i_dim={i_dim}, expected {want_packed} \
                 (wrong i_dim, or tensor isn't int4)",
                packed.len()
            );
        }
        Ok(Int4Matrix {
            packed,
            scale,
            o_dim,
            i_dim,
        })
    }

    /// Raw bytes of a tensor, straight out of the mmap (zero copy).
    pub fn bytes(&self, name: &str) -> Option<&[u8]> {
        let loc = self.index.get(name)?;
        self.shards.get(loc.shard)?.get(loc.begin..loc.end)
    }
}
