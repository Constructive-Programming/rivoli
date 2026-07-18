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

use anyhow::{Context, Result, bail, ensure};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::os::fd::{AsRawFd, RawFd};
use std::os::unix::fs::OpenOptionsExt;
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

/// A per-row **int8** weight matrix (the embedding table and lm_head, which the
/// snapshot keeps at int8 — one signed byte per weight + one f32 scale per row).
/// Obtained only via [`Snapshot::int8`], which validates the pairing/lengths so
/// a mismatched snapshot fails loudly instead of yielding stale/OOB logits.
#[derive(Debug, Clone, Copy)]
pub struct Int8Matrix<'a> {
    /// `o_dim` rows × `i_dim` signed bytes.
    pub packed: &'a [u8],
    /// `o_dim` little-endian f32 scales (raw bytes).
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

/// One raw header entry as written by safetensors.
#[derive(serde::Deserialize)]
struct RawTensor {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

/// One shard's mmap + open file (mmap serves header/dims/warm; the file serves
/// `pread` weight loads) plus the tensors located within it.
struct IndexedShard {
    mmap: Mmap,
    file: File,
    entries: Vec<(String, TensorLoc)>,
}

pub struct Snapshot {
    shards: Vec<Mmap>,
    files: Vec<File>, // same order as `shards`; buffered pread (resident build)
    odirect_fds: Vec<File>, // same order; O_DIRECT, for the io_uring cold streamer
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
        let mut files = Vec::with_capacity(paths.len());
        let mut odirect_fds = Vec::with_capacity(paths.len());
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
            files.push(shard.file);
            // A second fd opened O_DIRECT for the io_uring cold streamer (NVMe DMA
            // straight into the VMM slots, bypassing the page cache).
            let od = std::fs::OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_DIRECT)
                .open(path)
                .with_context(|| format!("open O_DIRECT {}", path.display()))?;
            odirect_fds.push(od);
        }
        Ok(Self {
            shards,
            files,
            odirect_fds,
            index,
        })
    }

    /// O_DIRECT read spec for a tensor: `(fd, file_begin, len)`. The fd + range the
    /// io_uring streamer submits (it does the block-alignment). `None` if missing.
    pub fn read_spec(&self, name: &str) -> Option<(RawFd, usize, usize)> {
        let loc = self.index.get(name)?;
        Some((
            self.odirect_fds.get(loc.shard)?.as_raw_fd(),
            loc.begin,
            loc.end - loc.begin,
        ))
    }

    fn index_shard(path: &Path) -> Result<IndexedShard> {
        let file = File::open(path).context("open shard")?;
        // SAFETY: the shard is a read-only model file; we never mutate the map
        // and it outlives no borrow past Snapshot's own lifetime.
        let mmap = unsafe { Mmap::map(&file) }.context("mmap shard")?;
        // `file` is kept open (returned below) for pread weight loads.
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
        Ok(IndexedShard {
            mmap,
            file,
            entries,
        })
    }

    /// `pread` a tensor's raw bytes directly into `dst` (must own `cap` bytes) —
    /// straight from the shard file into device-local host-mapped memory (a
    /// `VmmBuf`), no mmap fault and no H2D copy. If `evict`, drop the file's
    /// page-cache copy of the range afterward (`POSIX_FADV_DONTNEED`): resident
    /// weights are read once, so they shouldn't also pollute the cache the cold set
    /// needs (the pread equivalent of the old placement `madvise`). Cold loads pass
    /// `evict = false` — their file pages stay warm for re-hits.
    ///
    /// # Safety
    /// `dst` must point to at least `cap` writable bytes valid for this call.
    pub unsafe fn read_into(
        &self,
        name: &str,
        dst: *mut u8,
        cap: usize,
        evict: bool,
    ) -> Result<usize> {
        let loc = self
            .index
            .get(name)
            .with_context(|| format!("tensor {name} not found for pread"))?;
        let len = loc.end - loc.begin;
        ensure!(len <= cap, "read_into {name}: {len} bytes > dst cap {cap}");
        let fd = self.files[loc.shard].as_raw_fd();
        let mut done = 0usize;
        while done < len {
            // SAFETY: dst[done..len] is within the caller's `cap` bytes (checked);
            // pread writes at most len-done bytes there.
            let n = unsafe {
                libc::pread(
                    fd,
                    dst.add(done) as *mut libc::c_void,
                    len - done,
                    (loc.begin + done) as libc::off_t,
                )
            };
            ensure!(
                n > 0,
                "pread {name} (shard {} off {}): {}",
                loc.shard,
                loc.begin + done,
                std::io::Error::last_os_error()
            );
            done += n as usize;
        }
        if evict {
            // SAFETY: fd is a live open shard; advisory, never corrupts.
            unsafe {
                libc::posix_fadvise(
                    fd,
                    loc.begin as libc::off_t,
                    len as libc::off_t,
                    libc::POSIX_FADV_DONTNEED,
                );
            }
        }
        Ok(len)
    }

    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
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
        if !scale.len().is_multiple_of(4) {
            bail!("{sname}: {} scale bytes, not a multiple of 4", scale.len());
        }
        // NOTE: row_bytes = ceil(i_dim/2) can't distinguish i_dim=2k from 2k-1,
        // so a wrong ODD i_dim would pass the byte check. Harmless for GLM
        // (all dims even); revisit if an odd projection width ever appears.
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

    /// Locate an int8 weight matrix `W[o_dim, i_dim]` by base name (`<name>.weight`
    /// U8 + `<name>.weight.qs` F32). `o_dim` is derived from the scale count and
    /// cross-checked against the packed byte count (`o_dim * i_dim`, one byte per
    /// weight). Only constructor of [`Int8Matrix`].
    pub fn int8(&self, name: &str, i_dim: usize) -> Result<Int8Matrix<'_>> {
        let wname = format!("{name}.weight");
        let sname = format!("{name}.weight.qs");
        let wdt = self
            .index
            .get(&wname)
            .with_context(|| format!("int8 weight {wname} not found"))?
            .dtype;
        let sdt = self
            .index
            .get(&sname)
            .with_context(|| format!("int8 scale {sname} not found"))?
            .dtype;
        if wdt != Dtype::U8 {
            bail!("{wname} is {wdt:?}, expected U8 int8");
        }
        if sdt != Dtype::F32 {
            bail!("{sname} is {sdt:?}, expected F32 scale");
        }
        let packed = self.require(&wname)?;
        let scale = self.require(&sname)?;
        if !scale.len().is_multiple_of(4) {
            bail!("{sname}: {} scale bytes, not a multiple of 4", scale.len());
        }
        let o_dim = scale.len() / 4;
        if packed.len() != o_dim * i_dim {
            bail!(
                "{wname}: {} bytes for o_dim={o_dim} i_dim={i_dim}, expected {}",
                packed.len(),
                o_dim * i_dim
            );
        }
        Ok(Int8Matrix {
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
