//! Tokenizer: loads the snapshot's `tokenizer.json` (HuggingFace format) via
//! the `tokenizers` crate. Hand-rolling BPE would be the archetypal wheel to
//! not reinvent; this is a 19 MB trained vocab.

use anyhow::{Context, Result, anyhow};
use tracing::warn;

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
        // An empty eos means decode can never stop on an end token (runaway
        // generation) — surface a broken generation_config loudly, don't swallow.
        let eos = match Self::load_eos(snapshot_dir) {
            Ok(e) if !e.is_empty() => e,
            Ok(_) => {
                warn!("generation_config has no eos_token_id — decode won't stop on EOS");
                Vec::new()
            }
            Err(e) => {
                warn!("could not read eos ids ({e}) — decode won't stop on EOS");
                Vec::new()
            }
        };
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

    /// Decode a whole id sequence to text at once. Byte-level BPE splits one
    /// codepoint across several tokens, so decoding the complete sequence is
    /// what keeps multi-token characters (and any trailing partial one) intact —
    /// there is no incremental-flush footgun. Streaming detok belongs to server
    /// mode; it can be added there when it exists.
    pub fn decode_all(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, false)
            .map_err(|e| anyhow!("decode: {e}"))
    }
}
