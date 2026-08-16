//! The OpenAI chat-completion semantics, engine-free: an OpenAI `messages` array in, the
//! checkpoint template's turns out; a generation in, `reasoning_content` / `content` /
//! `tool_calls` out.
//!
//! Everything here is a pure function of JSON and text — no socket, no tokenizer, no
//! engine — which is what makes this the file the tests live in. The wire framing is
//! [`super::http`] and the request lifecycle is [`super`].
//!
//! Tool calling is the checkpoint's own, hand-ported from its `chat_template.jinja`
//! alongside the rest of the framing: declarations go out as the template's `# Tools`
//! system turn, calls come back as `<tool_call>name<arg_key>k</arg_key>…` and are parsed
//! into OpenAI `tool_calls`, and results return as `<|observation|><tool_response>`. The
//! renderer (`rivoli_artifact::tokenizer::tool_call_markup`) and the parser here are
//! deliberate mirrors — see [`parse_tool_calls`].

use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};

/// OpenAI `content` is either a string or an array of typed parts, and the chat UIs send
/// both shapes within one conversation.
///
/// A non-text part becomes the template's own `<reminder>` sentence rather than being
/// dropped. That is `visible_text()` in `chat_template.jinja` verbatim, and it is the
/// difference between the model answering "the image shows..." about an image it never
/// received and it saying it cannot see images — this engine is text-only, and a silent
/// drop makes the model confabulate.
fn content_text(c: Option<&Value>) -> String {
    fn part(p: &Value) -> Option<String> {
        if let Some(t) = p.as_str() {
            return Some(t.to_string());
        }
        match p.get("type").and_then(Value::as_str)? {
            "text" => p.get("text").and_then(Value::as_str).map(str::to_string),
            ty => {
                let media = ty.replace("_url", "").replace("input_", "");
                Some(format!(
                    "<reminder>You are unable to process this {media} because you don't have \
                     multi-modal input ability. Try different methods.</reminder>"
                ))
            }
        }
    }
    match c {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts.iter().filter_map(part).collect::<Vec<_>>().join(""),
        _ => String::new(),
    }
}

/// The roles the hand-ported template can frame.
const ROLES: [&str; 5] = ["system", "developer", "user", "assistant", "tool"];

/// Flatten an OpenAI `messages` array into the template's turns.
///
/// Two shapes do not survive one-message-per-turn and are folded here rather than in the
/// tokenizer, because both are facts about the OpenAI wire format and not about the
/// template:
/// - an assistant message carries `tool_calls` alongside (or instead of) its content, which
///   the template renders as markup INSIDE the assistant turn;
/// - consecutive `tool` results share ONE `<|observation|>` turn, so a run of them becomes a
///   single turn whose content is their concatenated `<tool_response>` blocks.
pub fn messages_to_turns(body: &Value) -> Result<Vec<(String, String)>> {
    let msgs = body
        .get("messages")
        .and_then(Value::as_array)
        .context("`messages` must be an array")?;
    ensure!(!msgs.is_empty(), "`messages` is empty");
    let mut turns: Vec<(String, String)> = Vec::with_capacity(msgs.len());
    for (i, m) in msgs.iter().enumerate() {
        let role = m.get("role").and_then(Value::as_str).unwrap_or("user");
        ensure!(
            ROLES.contains(&role),
            "messages[{i}].role is {role:?}; this server frames {ROLES:?} only"
        );
        let mut content = content_text(m.get("content"));
        match role {
            // A run of tool results is one observation turn. `last_mut` rather than a
            // look-ahead: the previous turn IS the run so far.
            "tool" => {
                let block = rivoli_artifact::tokenizer::tool_response_markup(&content);
                match turns.last_mut() {
                    Some((r, c)) if r == "observation" => c.push_str(&block),
                    _ => turns.push(("observation".to_string(), block)),
                }
                continue;
            }
            "assistant" => content.push_str(&rendered_calls(m, i)?),
            _ => {}
        }
        // `developer` is OpenAI's newer name for a system message; the template has no such
        // turn, so it frames as one.
        let role = if role == "developer" { "system" } else { role };
        turns.push((role.to_string(), content));
    }
    Ok(turns)
}

/// One assistant message's `tool_calls`, rendered back as the markup the template writes
/// them in — the tail of that turn's content, after whatever prose came with it.
fn rendered_calls(m: &Value, i: usize) -> Result<String> {
    let mut out = String::new();
    for (j, tc) in m
        .get("tool_calls")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .enumerate()
    {
        let f = tc.get("function").unwrap_or(tc);
        let name = f
            .get("name")
            .and_then(Value::as_str)
            .with_context(|| format!("messages[{i}].tool_calls[{j}] has no function name"))?;
        // OpenAI sends `arguments` as a JSON *string*; a client that sends an object
        // instead is accepted rather than argued with.
        let args = match f.get("arguments") {
            Some(Value::String(s)) => serde_json::from_str(s).unwrap_or_else(|_| json!({})),
            Some(v) => v.clone(),
            None => json!({}),
        };
        out.push_str(&rivoli_artifact::tokenizer::tool_call_markup(name, &args));
    }
    Ok(out)
}

const TOOL_OPEN: &str = "<tool_call>";
const TOOL_CLOSE: &str = "</tool_call>";

/// One of the template's paired markup delimiters.
///
/// A named pair rather than two loose `&str` parameters: `take(s, open, close)` is one
/// transposed argument away from silently scanning backwards, and the two are never useful
/// apart.
#[derive(Clone, Copy)]
struct Tag {
    open: &'static str,
    close: &'static str,
}

const ARG_KEY: Tag = Tag {
    open: "<arg_key>",
    close: "</arg_key>",
};
const ARG_VALUE: Tag = Tag {
    open: "<arg_value>",
    close: "</arg_value>",
};

/// `(inner, rest)` for the first `open`…`close` pair, or `None`.
fn take(s: &str, t: Tag) -> Option<(&str, &str)> {
    let i = s.find(t.open)? + t.open.len();
    let j = s[i..].find(t.close)?;
    Some((&s[i..i + j], &s[i + j + t.close.len()..]))
}

/// Pull the model's `<tool_call>` blocks out of a reply, returning the prose that was left
/// and the calls in OpenAI shape.
///
/// The inverse of `tokenizer::tool_call_markup`, and deliberately its mirror: an argument is
/// parsed as JSON and falls back to the raw string, because that is exactly how the renderer
/// decides between the two. `id` is derived from the completion id rather than random, so a
/// greedy engine stays reproducible request to request.
///
/// A block left unterminated by the token budget is still reported, with whatever arguments
/// completed — a truncated call the client can see beats a silent drop.
pub fn parse_tool_calls(text: &str, id: &str) -> (String, Vec<Value>) {
    let (mut prose, mut calls, mut rest) = (String::new(), Vec::new(), text);
    while let Some(i) = rest.find(TOOL_OPEN) {
        prose.push_str(&rest[..i]);
        let after = &rest[i + TOOL_OPEN.len()..];
        let (inner, tail) = match after.find(TOOL_CLOSE) {
            Some(j) => (&after[..j], &after[j + TOOL_CLOSE.len()..]),
            None => (after, ""),
        };
        let name_end = inner.find(ARG_KEY.open).unwrap_or(inner.len());
        calls.push(json!({
            "id": format!("call_{id}_{}", calls.len()),
            "type": "function",
            "function": {
                "name": inner[..name_end].trim(),
                // OpenAI's `arguments` is a JSON string, not an object.
                "arguments": Value::Object(parse_args(&inner[name_end..])).to_string(),
            }
        }));
        rest = tail;
    }
    prose.push_str(rest);
    (prose.trim().to_string(), calls)
}

/// The `<arg_key>k</arg_key><arg_value>v</arg_value>` run inside one call block.
///
/// A value is parsed as JSON and falls back to the raw string, mirroring the renderer's
/// `v | tojson if v is not string else v`: that fallback is what makes the pair round-trip
/// rather than turning `Paris` into a parse error.
fn parse_args(mut cursor: &str) -> serde_json::Map<String, Value> {
    let mut args = serde_json::Map::new();
    while let Some((key, r)) = take(cursor, ARG_KEY) {
        let Some((raw, r2)) = take(r, ARG_VALUE) else {
            break;
        };
        let v = serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string()));
        args.insert(key.to_string(), v);
        cursor = r2;
    }
    args
}

/// The part of `text` that is safe to stream as `content` right now.
///
/// Everything before the first `<tool_call>` and not a byte more — a tool call is a protocol
/// message, not prose, and it goes out as a structured delta at the end instead. A trailing
/// PARTIAL marker is held back too: mid-generation the text can end `…<tool_ca`, and emitting
/// that would leak a fragment which the next token turns into a marker, after which `content`
/// would have to shrink. A delta stream cannot express shrinking.
pub fn streamable(text: &str) -> &str {
    if let Some(i) = text.find(TOOL_OPEN) {
        return &text[..i];
    }
    // Longest suffix that is a proper prefix of the marker. ASCII, so a match can never
    // land mid-codepoint.
    for k in (1..TOOL_OPEN.len().min(text.len() + 1)).rev() {
        if text.ends_with(&TOOL_OPEN[..k]) {
            return &text[..text.len() - k];
        }
    }
    text
}

/// Split a generation into `(reasoning, content)`.
///
/// With thinking ON the prompt ends at an OPEN `<think>`, so the model emits its reasoning
/// first and closes it — everything after `</think>` is the answer. With thinking off the
/// prompt already closed it and no tags appear at all, which is why this needs to be told
/// which mode it is in rather than guessing from the text.
///
/// A generation that hits the token budget mid-reasoning has no close, so it is all
/// reasoning and no content. Reporting that honestly is the point: the alternative is
/// presenting a half-finished train of thought as the answer.
pub fn split_think(full: &str, thinking: bool) -> (&str, &str) {
    if !thinking {
        return ("", full);
    }
    match full.split_once("</think>") {
        Some((reasoning, content)) => (reasoning, content.trim_start()),
        None => (full, ""),
    }
}

/// The new text a token added, given everything already sent — `None` when it added
/// nothing emittable yet.
///
/// Byte-level BPE splits one codepoint across several tokens, so decoding a PREFIX of the
/// generation can end in U+FFFD: a stub the next token completes into a real character.
/// Emitting it would leave a permanent replacement char in the stream, so it is held back
/// and the next decode supersedes it. `Tokenizer::decode_all` names this the streaming
/// detok footgun and says server mode is where it gets paid. This is that payment.
pub fn delta<'a>(sent: &str, full: &'a str) -> Option<&'a str> {
    let stable = full.trim_end_matches('\u{FFFD}');
    // `strip_prefix` rather than slicing at `sent.len()`: if a decode ever fails to extend
    // what we already sent, emitting nothing is wrong but harmless, and a panic is not.
    stable.strip_prefix(sent).filter(|d| !d.is_empty())
}

/// `tool_calls` outranks `stop` — an agent loop branches on this field, and a reply
/// carrying calls but reporting `stop` reads as "the model is done talking to you",
/// which is the opposite of what it means. `length` still wins over both: a call cut
/// off by the budget may be incomplete, and saying `tool_calls` would assert it is not.
///
/// EOS is the only way a decode ends short of its budget, so reaching it IS `length`.
pub fn stop_reason(calls: &[Value], generated: usize, ngen: usize) -> &'static str {
    match (generated >= ngen, calls.is_empty()) {
        (true, _) => "length",
        (false, false) => "tool_calls",
        (false, true) => "stop",
    }
}

/// Who is answering, repeated verbatim on every frame of one reply.
///
/// A named bundle rather than three parameters threaded through the emitters: a client
/// reassembling a stream keys on `id`, so a frame that carried a different one would be a
/// different completion as far as it is concerned, and three loose arguments are three
/// chances to hand one emitter a stale copy. (`chunk` took all three plus its payload and
/// was the file's one Excess-Arguments finding, CodeScene 2026-08-16 — this is the
/// abstraction it was naming.)
pub struct Completion {
    pub id: String,
    /// Unix seconds, echoed on every frame and on the final body.
    pub created: u64,
    /// The model name the client asked for, echoed back — not necessarily the one loaded.
    pub model: String,
}

impl Completion {
    /// One streamed frame: a `delta` to merge, or a `finish_reason` to stop on.
    pub fn chunk(&self, delta: Value, finish: Option<&str>) -> Value {
        json!({"id": self.id, "object": "chat.completion.chunk", "created": self.created,
               "model": self.model,
               "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})
    }
}

pub fn err_body(msg: &str) -> Value {
    json!({"error": {"message": msg, "type": "invalid_request_error"}})
}

/// Unix seconds, which is what OpenAI's `created` carries on both a completion and a model
/// listing. A clock before the epoch is reported as 0 rather than refused: `created` is
/// informational, and no request should fail on it.
pub fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)] // a test harness that cannot parse its own fixture should panic
mod tests {
    use super::*;

    #[test]
    fn flattens_both_content_shapes() {
        let b: Value = serde_json::from_str(
            r#"{"messages":[{"role":"system","content":"be terse"},
                            {"role":"user","content":[{"type":"text","text":"hi"},
                                                      {"type":"image_url","image_url":{"url":"x"}},
                                                      {"type":"text","text":" there"}]}]}"#,
        )
        .unwrap();
        // The image becomes the template's own reminder rather than vanishing. Dropping it
        // used to be the behaviour and it made the model describe images it never got.
        assert_eq!(
            messages_to_turns(&b).unwrap(),
            vec![
                ("system".to_string(), "be terse".to_string()),
                (
                    "user".to_string(),
                    "hi<reminder>You are unable to process this image because you don't have \
                     multi-modal input ability. Try different methods.</reminder> there"
                        .to_string()
                ),
            ]
        );
    }

    #[test]
    fn no_messages_is_an_error_not_an_empty_prompt() {
        assert!(messages_to_turns(&json!({"messages": []})).is_err());
        assert!(messages_to_turns(&json!({"prompt": "hi"})).is_err());
    }

    #[test]
    fn developer_is_a_system_turn_and_an_unknown_role_is_refused() {
        // OpenAI renamed `system` to `developer`; the template has only the one turn token.
        assert_eq!(
            messages_to_turns(&json!({"messages": [{"role": "developer", "content": "x"}]}))
                .unwrap(),
            vec![("system".to_string(), "x".to_string())]
        );
        let e = messages_to_turns(&json!({"messages": [{"role": "wizard", "content": "x"}]}))
            .unwrap_err()
            .to_string();
        assert!(e.contains("wizard"), "{e}");
    }

    #[test]
    fn a_tool_round_trip_folds_into_the_template_turns() {
        let b = json!({"messages": [
            {"role": "user", "content": "weather in Paris and Rome?"},
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "a", "type": "function",
                 "function": {"name": "wx", "arguments": "{\"city\":\"Paris\"}"}},
                {"id": "b", "type": "function",
                 "function": {"name": "wx", "arguments": "{\"city\":\"Rome\"}"}}]},
            {"role": "tool", "tool_call_id": "a", "content": "18C"},
            {"role": "tool", "tool_call_id": "b", "content": "24C"},
        ]});
        assert_eq!(
            messages_to_turns(&b).unwrap(),
            vec![
                ("user".to_string(), "weather in Paris and Rome?".to_string()),
                // Calls render INSIDE the assistant turn, in order, after its (empty) prose.
                (
                    "assistant".to_string(),
                    "<tool_call>wx<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>\
                     <tool_call>wx<arg_key>city</arg_key><arg_value>Rome</arg_value></tool_call>"
                        .to_string()
                ),
                // BOTH results share ONE observation turn — the template opens it once per
                // consecutive run, not once per result.
                (
                    "observation".to_string(),
                    "<tool_response>18C</tool_response><tool_response>24C</tool_response>"
                        .to_string()
                ),
            ]
        );
    }

    #[test]
    fn parse_tool_calls_mirrors_the_renderer() {
        // A string argument is rendered RAW and everything else as JSON, so the parse has to
        // try JSON first and fall back — that is what makes the two round-trip.
        let (prose, calls) = parse_tool_calls(
            "Let me look.<tool_call>wx<arg_key>city</arg_key><arg_value>Paris</arg_value>\
             <arg_key>days</arg_key><arg_value>3</arg_value></tool_call>",
            "X",
        );
        assert_eq!(prose, "Let me look.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["function"]["name"], "wx");
        assert_eq!(calls[0]["id"], "call_X_0");
        // `arguments` is a JSON STRING in the OpenAI shape, not an object.
        let args: Value =
            serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args, json!({"city": "Paris", "days": 3}));

        // Two calls, no prose.
        let (prose, calls) = parse_tool_calls(
            "<tool_call>a</tool_call><tool_call>b<arg_key>k</arg_key><arg_value>v</arg_value>\
             </tool_call>",
            "X",
        );
        assert!(prose.is_empty());
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[1]["id"], "call_X_1");
        assert_eq!(calls[0]["function"]["arguments"], "{}");

        // Budget ran out mid-call: report the truncated call rather than dropping it.
        let (_, calls) = parse_tool_calls("<tool_call>wx<arg_key>city</arg_key>", "X");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["function"]["name"], "wx");
    }

    /// The mirror claim stated as a round trip rather than as two hand-written literals:
    /// the renderer is the tokenizer's, so a change on either side that broke the pairing
    /// would pass both files' own tests and fail here.
    #[test]
    fn the_renderer_and_the_parser_round_trip() {
        let args = json!({"city": "Paris", "days": 3, "exact": false});
        let markup = rivoli_artifact::tokenizer::tool_call_markup("wx", &args);
        let (prose, calls) = parse_tool_calls(&markup, "R");
        assert!(prose.is_empty(), "{prose:?}");
        assert_eq!(calls.len(), 1);
        let back: Value =
            serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(back, args);
    }

    #[test]
    fn streamable_holds_back_anything_that_could_become_a_tool_call() {
        assert_eq!(streamable("plain prose"), "plain prose");
        assert_eq!(streamable("before<tool_call>wx"), "before");
        // A partial marker must not leak: the next token completes it, and `content` cannot
        // shrink once sent.
        assert_eq!(streamable("ok <tool_c"), "ok ");
        assert_eq!(streamable("ok <"), "ok ");
        // A lone `<` in prose is held one step and emitted as soon as it cannot be a marker.
        assert_eq!(streamable("a <b"), "a <b");
        assert_eq!(streamable(""), "");
    }

    #[test]
    fn split_think_needs_to_be_told_which_mode_it_is_in() {
        // Thinking off: the prompt already closed <think>, so nothing is reasoning. Guessing
        // from the text would make a whole answer disappear into the reasoning channel.
        assert_eq!(
            split_think("the sky is blue", false),
            ("", "the sky is blue")
        );
        // Thinking on: the open <think> is in the PROMPT, so the generation starts inside it.
        assert_eq!(
            split_think("hmm, scattering</think>The sky is blue.", true),
            ("hmm, scattering", "The sky is blue.")
        );
        // Budget ran out mid-reasoning — all reasoning, no answer, and say so rather than
        // presenting a half-finished train of thought as the reply.
        assert_eq!(split_think("hmm, scatter", true), ("hmm, scatter", ""));
    }

    #[test]
    fn delta_holds_back_a_split_codepoint_until_it_completes() {
        // A byte-level BPE token can end mid-codepoint: the prefix decodes to a lone
        // U+FFFD stub, and the next token completes it. Emitting the stub would leave a
        // replacement character in the stream forever.
        assert_eq!(delta("ok ", "ok \u{FFFD}"), None);
        assert_eq!(delta("ok ", "ok é"), Some("é"));
        assert_eq!(delta("ok é", "ok é!"), Some("!"));
        assert_eq!(delta("ok", "ok"), None);
        // Defensive: a decode that does not extend what was sent emits nothing, not a panic.
        assert_eq!(delta("ok", "different"), None);
    }

    /// The contract an agent loop branches on. `length` outranks `tool_calls` because a
    /// call cut off by the budget may be incomplete.
    #[test]
    fn stop_reason_ranks_length_over_calls_over_stop() {
        let call = vec![json!({"id": "call_X_0"})];
        assert_eq!(stop_reason(&[], 8, 8), "length");
        assert_eq!(stop_reason(&call, 8, 8), "length");
        assert_eq!(stop_reason(&call, 3, 8), "tool_calls");
        assert_eq!(stop_reason(&[], 3, 8), "stop");
    }
}
