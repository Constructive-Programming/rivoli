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
        let v = super::serde_value::parse(s)?;
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

/// Minimal JSON value good enough for a safetensors header (objects, arrays,
/// strings, numbers). Avoids pulling serde_json for one flat parse.
mod serde_value {
    use anyhow::{Result, bail};
    use std::collections::BTreeMap;

    pub enum Value {
        Object(BTreeMap<String, Value>),
        Array(Vec<Value>),
        Str(String),
        Num(f64),
        Bool,
        Null,
    }

    impl Value {
        pub fn as_object(&self) -> Option<&BTreeMap<String, Value>> {
            match self {
                Value::Object(m) => Some(m),
                _ => None,
            }
        }
        pub fn as_array(&self) -> Option<&Vec<Value>> {
            match self {
                Value::Array(a) => Some(a),
                _ => None,
            }
        }
        pub fn as_str(&self) -> Option<&str> {
            match self {
                Value::Str(s) => Some(s),
                _ => None,
            }
        }
        pub fn as_u64(&self) -> Option<u64> {
            match self {
                Value::Num(n) => Some(*n as u64),
                _ => None,
            }
        }
    }

    pub fn parse(s: &str) -> Result<Value> {
        let mut p = Parser {
            b: s.as_bytes(),
            i: 0,
        };
        p.ws();
        let v = p.value()?;
        Ok(v)
    }

    struct Parser<'a> {
        b: &'a [u8],
        i: usize,
    }

    impl Parser<'_> {
        fn ws(&mut self) {
            while self.i < self.b.len() && self.b[self.i].is_ascii_whitespace() {
                self.i += 1;
            }
        }
        fn value(&mut self) -> Result<Value> {
            self.ws();
            match self.b.get(self.i) {
                Some(b'{') => self.object(),
                Some(b'[') => self.array(),
                Some(b'"') => Ok(Value::Str(self.string()?)),
                Some(b't') => self.lit("true", Value::Bool),
                Some(b'f') => self.lit("false", Value::Bool),
                Some(b'n') => self.lit("null", Value::Null),
                Some(_) => self.number(),
                None => bail!("unexpected end of json"),
            }
        }
        fn lit(&mut self, kw: &str, v: Value) -> Result<Value> {
            if self.b[self.i..].starts_with(kw.as_bytes()) {
                self.i += kw.len();
                Ok(v)
            } else {
                bail!("bad literal at {}", self.i)
            }
        }
        fn object(&mut self) -> Result<Value> {
            self.i += 1; // {
            let mut m = std::collections::BTreeMap::new();
            self.ws();
            if self.b.get(self.i) == Some(&b'}') {
                self.i += 1;
                return Ok(Value::Object(m));
            }
            loop {
                self.ws();
                let k = self.string()?;
                self.ws();
                if self.b.get(self.i) != Some(&b':') {
                    bail!("expected ':' at {}", self.i);
                }
                self.i += 1;
                let v = self.value()?;
                m.insert(k, v);
                self.ws();
                match self.b.get(self.i) {
                    Some(b',') => self.i += 1,
                    Some(b'}') => {
                        self.i += 1;
                        break;
                    }
                    _ => bail!("expected ',' or '}}' at {}", self.i),
                }
            }
            Ok(Value::Object(m))
        }
        fn array(&mut self) -> Result<Value> {
            self.i += 1; // [
            let mut a = Vec::new();
            self.ws();
            if self.b.get(self.i) == Some(&b']') {
                self.i += 1;
                return Ok(Value::Array(a));
            }
            loop {
                a.push(self.value()?);
                self.ws();
                match self.b.get(self.i) {
                    Some(b',') => self.i += 1,
                    Some(b']') => {
                        self.i += 1;
                        break;
                    }
                    _ => bail!("expected ',' or ']' at {}", self.i),
                }
            }
            Ok(Value::Array(a))
        }
        fn string(&mut self) -> Result<String> {
            if self.b.get(self.i) != Some(&b'"') {
                bail!("expected '\"' at {}", self.i);
            }
            self.i += 1;
            let mut out = String::new();
            while let Some(&c) = self.b.get(self.i) {
                self.i += 1;
                match c {
                    b'"' => return Ok(out),
                    b'\\' => {
                        let e = *self.b.get(self.i).unwrap_or(&b'"');
                        self.i += 1;
                        out.push(match e {
                            b'n' => '\n',
                            b't' => '\t',
                            b'r' => '\r',
                            b'"' => '"',
                            b'\\' => '\\',
                            b'/' => '/',
                            other => other as char,
                        });
                    }
                    _ => out.push(c as char),
                }
            }
            bail!("unterminated string")
        }
        fn number(&mut self) -> Result<Value> {
            let start = self.i;
            while let Some(&c) = self.b.get(self.i) {
                if c == b'-'
                    || c == b'+'
                    || c == b'.'
                    || c == b'e'
                    || c == b'E'
                    || c.is_ascii_digit()
                {
                    self.i += 1;
                } else {
                    break;
                }
            }
            let s = std::str::from_utf8(&self.b[start..self.i])?;
            Ok(Value::Num(s.parse()?))
        }
    }
}
