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

    /// GLM turn framing: `[gMASK] <sop> <|user|> \n {text} <|assistant|> \n`.
    ///
    /// **Without this the model can never stop.** Two of its three EOS ids are turn
    /// boundaries (`<|user|>` 154827, `<|observation|>` 154829) and the third is
    /// `<|endoftext|>` — all of which an instruct model emits at the end of an ASSISTANT
    /// TURN. Fed raw text it is doing document continuation, is never in a turn, and has
    /// no reason to emit any of them: across 56 benchmark runs not one terminated
    /// naturally, every one ran to its token limit. Forced to keep writing past the end
    /// of its answer it drifts into list scaffolding and then loops
    /// (`**Memory Product.**` x329) — which is what invalidated every matrix cell above
    /// 2048 tokens. See docs/measurement/benchmarks.md's retraction.
    ///
    /// Built from token IDS rather than by encoding the literal template text, so it does
    /// not depend on the tokenizer choosing to match `[gMASK]` as a special token inside a
    /// string. No Jinja either: the checkpoint ships no `chat_template`, and GLM's turn
    /// structure is four tokens of framing.
    pub fn encode_chat(&self, text: &str) -> Result<Vec<u32>> {
        // Missing any of these means an artifact whose tokenizer predates the chat
        // tokens; fall back rather than fail, but say so — silent raw encoding is the
        // bug this function exists to fix.
        let ids: Option<Vec<u32>> = ["[gMASK]", "<sop>", "<|user|>", "<|assistant|>"]
            .iter()
            .map(|t| self.inner.token_to_id(t))
            .collect();
        let Some(sp) = ids else {
            warn!(
                "tokenizer lacks GLM chat tokens ([gMASK]/<sop>/<|user|>/<|assistant|>) — \
                 encoding the prompt RAW. Decode will not stop on EOS."
            );
            return self.encode(text);
        };
        let nl = self.encode("\n")?;
        let body = self.encode(text)?;
        let mut out = Vec::with_capacity(body.len() + nl.len() * 2 + 4);
        out.push(sp[0]); // [gMASK]
        out.push(sp[1]); // <sop>
        out.push(sp[2]); // <|user|>
        out.extend_from_slice(&nl);
        out.extend_from_slice(&body);
        out.push(sp[3]); // <|assistant|>
        out.extend_from_slice(&nl);
        Ok(out)
    }

    /// Encode prompt text to token ids with NO special tokens — raw continuation.
    /// Used by `--ppl` (which scores a fixed corpus, not a chat turn) and by
    /// `--raw-prompt` for reproducing pre-templating benchmark numbers.
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
