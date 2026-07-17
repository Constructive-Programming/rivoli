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

/// Tensor element type, validated at index time so downstream code matches on
/// an enum rather than a raw string. Only the dtypes this model uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F32,
    U8,
    I8,
}

impl Dtype {
    fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "F32" => Dtype::F32,
            "U8" => Dtype::U8,
            "I8" => Dtype::I8,
            other => bail!("unsupported tensor dtype {other:?}"),
        })
    }
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
                index.insert(name, TensorLoc { shard: si, ..loc });
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

    /// Raw bytes of a tensor, straight out of the mmap (zero copy).
    pub fn bytes(&self, name: &str) -> Option<&[u8]> {
        let loc = self.index.get(name)?;
        self.shards.get(loc.shard)?.get(loc.begin..loc.end)
    }
}
