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

/// Where one tensor's bytes live: which mmap'd shard and the byte range within.
#[derive(Debug, Clone)]
pub struct TensorLoc {
    pub shard: usize,
    pub begin: usize,
    pub end: usize,
    pub dtype: String,
    pub shape: Vec<usize>,
}

impl TensorLoc {
    pub fn nbytes(&self) -> usize {
        self.end - self.begin
    }
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
        let header: serde_json_header::Header = serde_json_header::parse(&mmap[8..8 + hlen])
            .context("parse safetensors header json")?;
        let data_start = 8 + hlen;

        let mut entries = Vec::with_capacity(header.tensors.len());
        for (name, t) in header.tensors {
            if name == "__metadata__" {
                continue;
            }
            entries.push((
                name,
                TensorLoc {
                    shard: 0, // set by caller
                    begin: data_start + t.begin,
                    end: data_start + t.end,
                    dtype: t.dtype,
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

/// Tiny hand-rolled safetensors-header parser: pulls exactly the fields we
/// need without adding a serde_json dependency for one flat object of objects.
mod serde_json_header {
    use anyhow::{Result, bail};

    pub struct Tensor {
        pub dtype: String,
        pub shape: Vec<usize>,
        pub begin: usize,
        pub end: usize,
    }

    pub struct Header {
        pub tensors: Vec<(String, Tensor)>,
    }

    /// The header is a single JSON object: {"name":{"dtype":..,"shape":[..],
    /// "data_offsets":[b,e]}, ..., "__metadata__":{...}}. We parse structurally
    /// (no general JSON) — string keys, then per-tensor the three fields.
    pub fn parse(bytes: &[u8]) -> Result<Header> {
        let s = std::str::from_utf8(bytes)?;
        let v = crate::json::parse(s)?;
        let obj = v
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("header not an object"))?;
        let mut tensors = Vec::with_capacity(obj.len());
        for (name, tv) in obj {
            if name == "__metadata__" {
                continue;
            }
            let t = tv
                .as_object()
                .ok_or_else(|| anyhow::anyhow!("tensor {name} not object"))?;
            let dtype = t
                .get("dtype")
                .and_then(|d| d.as_str())
                .ok_or_else(|| anyhow::anyhow!("tensor {name} missing dtype"))?
                .to_string();
            let shape = t
                .get("shape")
                .and_then(|s| s.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|n| n.as_u64().map(|n| n as usize))
                        .collect()
                })
                .unwrap_or_default();
            let off = t
                .get("data_offsets")
                .and_then(|o| o.as_array())
                .ok_or_else(|| anyhow::anyhow!("tensor {name} missing data_offsets"))?;
            if off.len() != 2 {
                bail!("tensor {name} data_offsets not [begin,end]");
            }
            let begin = off[0].as_u64().unwrap_or(0) as usize;
            let end = off[1].as_u64().unwrap_or(0) as usize;
            tensors.push((
                name.clone(),
                Tensor {
                    dtype,
                    shape,
                    begin,
                    end,
                },
            ));
        }
        Ok(Header { tensors })
    }
}
