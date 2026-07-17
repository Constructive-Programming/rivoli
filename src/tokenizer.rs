//! Tokenizer: loads the snapshot's `tokenizer.json` (HuggingFace format) via
//! the `tokenizers` crate. Hand-rolling BPE would be the archetypal wheel to
//! not reinvent; this is a 19 MB trained vocab.

use anyhow::{Context, Result, anyhow};

pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
    /// End-of-sequence ids from generation_config (any one stops decode).
    pub eos: Vec<u32>,
}

impl Tokenizer {
    pub fn load(snapshot_dir: &str) -> Result<Self> {
        let path = format!("{snapshot_dir}/tokenizer.json");
        let inner =
            tokenizers::Tokenizer::from_file(&path).map_err(|e| anyhow!("load {path}: {e}"))?;
        let eos = Self::load_eos(snapshot_dir).unwrap_or_default();
        Ok(Self { inner, eos })
    }

    fn load_eos(dir: &str) -> Result<Vec<u32>> {
        let text = std::fs::read_to_string(format!("{dir}/generation_config.json"))?;
        let v: serde_json::Value = serde_json::from_str(&text)?;
        let e = v.get("eos_token_id").context("no eos_token_id")?;
        // Either a single int or an array of ints.
        let ids = match e {
            serde_json::Value::Array(a) => a
                .iter()
                .filter_map(|x| x.as_u64().map(|n| n as u32))
                .collect(),
            other => other.as_u64().map(|n| vec![n as u32]).unwrap_or_default(),
        };
        Ok(ids)
    }

    /// Encode prompt text to token ids (no special tokens added — GLM chat
    /// templating is a later concern; M1 just needs a coherent continuation).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow!("encode: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Decode a single token id to its text piece.
    pub fn decode1(&self, id: u32) -> Result<String> {
        self.inner
            .decode(&[id], false)
            .map_err(|e| anyhow!("decode: {e}"))
    }

    pub fn is_eos(&self, id: u32) -> bool {
        self.eos.contains(&id)
    }
}
