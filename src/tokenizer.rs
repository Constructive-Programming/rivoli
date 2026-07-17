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

    /// A streaming decoder for the generation loop. Byte-level BPE routinely
    /// splits one codepoint across several tokens, so decoding ids in isolation
    /// mangles multi-token characters; this holds the running id buffer and
    /// emits only the newly-stable text suffix per step.
    pub fn decoder(&self) -> Decoder<'_> {
        Decoder {
            tok: self,
            ids: Vec::new(),
            emitted: 0,
        }
    }

    pub fn is_eos(&self, id: u32) -> bool {
        self.eos.contains(&id)
    }
}

/// Incremental detokenizer. Re-decodes the growing id buffer and returns the
/// character delta beyond what was already emitted — so a codepoint split
/// across tokens surfaces once, whole, when its last token arrives.
pub struct Decoder<'a> {
    tok: &'a Tokenizer,
    ids: Vec<u32>,
    emitted: usize,
}

impl Decoder<'_> {
    /// Feed one generated id; returns the new text to append (possibly empty
    /// while a multi-token codepoint is still incomplete).
    pub fn step(&mut self, id: u32) -> Result<String> {
        self.ids.push(id);
        let text = self
            .tok
            .inner
            .decode(&self.ids, false)
            .map_err(|e| anyhow!("decode: {e}"))?;
        // Emit the stable prefix (everything up to a trailing replacement char,
        // which marks an incomplete codepoint) beyond what we've already sent.
        let stable = text.trim_end_matches('\u{FFFD}');
        let stable_chars = stable.chars().count();
        if stable_chars <= self.emitted {
            return Ok(String::new());
        }
        let out: String = stable.chars().skip(self.emitted).collect();
        self.emitted = stable_chars;
        Ok(out)
    }
}
