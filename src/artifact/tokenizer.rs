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

/// Thinking control for [`Tokenizer::encode_chat_turns`], named after the parameters of the
/// same names in the checkpoint's `chat_template.jinja`.
///
/// Thinking is a PREFILL, not a flag the model reads: the generation prompt ends at an open
/// `<think>` and the model reasons until it emits `</think>`, or ends at `<think></think>`
/// and it answers straight away. The vocabulary has a `/nothink` token (154851) but the
/// template never emits it, so neither does this.
#[derive(Debug, Clone, Copy, Default)]
pub struct ChatOpts<'a> {
    /// Leave `<think>` open so the model reasons first.
    ///
    /// **The template's default is `true`; this struct's is `false`, deliberately.** At
    /// ~2.7 tok/s a reasoning block is tens of seconds of silence before the first word of
    /// the answer, and an OpenAI client has no way to ask for it to stop. `--think` and the
    /// request's `enable_thinking` turn it back on per server and per request.
    pub thinking: bool,
    /// `Some("high")` renders "Reasoning Effort: High"; everything else renders "Max",
    /// which is the template's own `capitalize` of its default. Ignored unless `thinking`.
    pub reasoning_effort: Option<&'a str>,
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
    /// string. No Jinja engine either — see [`Tokenizer::encode_chat_turns`], which is a
    /// hand-port of the checkpoint's `chat_template.jinja` down to the byte.
    pub fn encode_chat(&self, text: &str) -> Result<Vec<u32>> {
        self.encode_chat_turns(&[("user", text)], &ChatOpts::default())
    }

    /// The same framing over a MULTI-TURN conversation, which is what an OpenAI `messages`
    /// array is. **This is a hand-port of the checkpoint's `chat_template.jinja`, and it is
    /// meant to match it byte for byte:**
    ///
    /// ```text
    /// [gMASK] <sop>
    /// [<|system|> "Reasoning Effort: High|Max"]      -- only when thinking
    /// <|role|> {content}                             -- per turn, NO separator
    /// <|assistant|> <think> </think> {content}       -- assistant turns clear reasoning
    /// <|assistant|> <think> [</think>]               -- the generation prompt
    /// ```
    ///
    /// > **CORRECTED 2026-08-01. There is no newline after the role token, and there never
    /// > was.** This function used to emit `<|role|> \n {content}` and end the prompt at
    /// > `<|assistant|> \n`, which is the GLM-4 template, not this checkpoint's. Every
    /// > benchmark before that date was measured one token off-template per turn and with
    /// > the thinking prefill missing entirely, so its text is not comparable with anything
    /// > measured after. The source of truth is `chat_template.jinja` in the fp8 checkpoint
    /// > (`manifest.json`'s `i4_source.src`); the converted artifact does not carry it,
    /// > which is why this drifted unnoticed.
    ///
    /// The `<think>` prefill is the whole of thinking control — see [`ChatOpts`]. Note the
    /// template never emits the `/nothink` token, despite the vocab having one.
    ///
    /// Roles map to GLM's own turn tokens (`system`/`user`/`assistant`/`observation`); an
    /// unknown role is framed as `user`, since the alternative — dropping the turn — loses
    /// content silently.
    pub fn encode_chat_turns(&self, turns: &[(&str, &str)], opts: &ChatOpts) -> Result<Vec<u32>> {
        // Missing any of these means an artifact whose tokenizer predates the chat tokens;
        // fall back rather than fail, but say so — silent raw encoding is the bug this
        // function exists to fix. `<think>`/`</think>` are in the same lookup because the
        // generation prompt cannot be built without them.
        let ids: Option<Vec<u32>> = [
            "[gMASK]",
            "<sop>",
            "<|user|>",
            "<|assistant|>",
            "<think>",
            "</think>",
        ]
        .iter()
        .map(|t| self.inner.token_to_id(t))
        .collect();
        let Some(sp) = ids else {
            warn!(
                "tokenizer lacks the GLM chat tokens ([gMASK]/<sop>/<|user|>/<|assistant|>/\
                 <think>) — encoding the prompt RAW. Decode will not stop on EOS."
            );
            return self.encode(&turns.iter().map(|(_, t)| *t).collect::<Vec<_>>().join("\n"));
        };
        let (user, assistant, think_open, think_close) = (sp[2], sp[3], sp[4], sp[5]);
        let mut out = vec![sp[0], sp[1]]; // [gMASK] <sop>

        // The template emits this whenever thinking is on, BEFORE the conversation, so the
        // model saw it there in training. `capitalize` of the template's own default makes
        // anything that is not "high" into "Max" — including "low" and "medium", which is
        // the template's behaviour and not a shortcut taken here.
        if opts.thinking && let Some(system) = self.inner.token_to_id("<|system|>") {
            let effort = if opts.reasoning_effort == Some("high") { "High" } else { "Max" };
            out.push(system);
            out.extend_from_slice(&self.encode(&format!("Reasoning Effort: {effort}"))?);
        }

        for (role, text) in turns {
            let header = match *role {
                "user" => user,
                "assistant" => assistant,
                // Only looked up when actually used: an artifact can carry the tokens above
                // without these, and a `system` message is optional.
                other => match self.inner.token_to_id(&format!("<|{other}|>")) {
                    Some(id) => id,
                    None => {
                        warn!("tokenizer has no <|{other}|> turn token — framing it as user");
                        user
                    }
                },
            };
            out.push(header);
            if *role == "assistant" {
                // History carries NO reasoning. The template keeps it only for assistant
                // turns after the last user message (a continuation), so for a request that
                // ends in a user turn — every request a chat client makes — this is all of
                // them. `.trim()` matches the template's `content.strip()`.
                out.push(think_open);
                out.push(think_close);
                out.extend_from_slice(&self.encode(text.trim())?);
            } else {
                out.extend_from_slice(&self.encode(text)?);
            }
        }

        // The generation prompt. An OPEN `<think>` is the template's default and makes the
        // model reason before answering; closing it immediately is how thinking is turned
        // off. Either way the model is inside the structure it was trained on.
        out.push(assistant);
        out.push(think_open);
        if !opts.thinking {
            out.push(think_close);
        }
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
