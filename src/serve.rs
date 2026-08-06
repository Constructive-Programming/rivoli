//! OpenAI-compatible HTTP server (`--port`) — the shape llama-swap, Open WebUI and every
//! OpenAI client already speak, so rivoli becomes a swappable backend without a wrapper
//! process in front of it.
//!
//! **Hand-rolled HTTP/1.1 over `std::net`, no HTTP crate, no async runtime, one request at
//! a time.** That is not a shortcut around a missing feature: this engine decodes at ~3
//! tok/s on a GPU that CLAUDE.md requires be sole-tenant, so concurrency here could only
//! queue what the device already serialises. Every response carries `Connection: close` —
//! one request per connection — which removes the keep-alive state machine entirely and is
//! what a reverse proxy in front of us opens anyway.
//!
//! **This is an inference backend, not a chat product.** Open WebUI and the Hermes agent
//! already own the conversation surface; everything here exists to be called BY them. That
//! is why thinking and tool calling are expressed as the protocol fields those clients
//! already send and read — `enable_thinking`/`reasoning_content`, `tools`/`tool_calls` —
//! and nothing here renders, orchestrates or loops.
//!
//! Tool calling is the checkpoint's own, hand-ported from its `chat_template.jinja`
//! alongside the rest of the framing: declarations go out as the template's `# Tools`
//! system turn, calls come back as `<tool_call>name<arg_key>k</arg_key>…` and are parsed
//! into OpenAI `tool_calls`, and results return as `<|observation|><tool_response>`. The
//! renderer and the parser are deliberate mirrors — see `parse_tool_calls`.
//!
//! Deliberately absent, so nobody goes looking:
//! - **Sampling.** The engine is greedy argmax and every number in `docs/measurement/benchmarks.md` is
//!   measured that way. `temperature`/`top_p` are accepted and IGNORED, with one warning
//!   per process — honouring them is not a server-side change, and dropping them silently
//!   would leave a client believing its own determinism story.
//! - **Forcing a call.** `tool_choice` accepts `"auto"` and `"none"` (which drops the
//!   declarations, genuinely preventing calls) and REFUSES `"required"` or a named
//!   function: nothing here can constrain decoding, and answering prose to a client that
//!   demanded a call would look like compliance.
//! - **`/v1/completions`.** A raw (unframed) prompt leaves the model outside an assistant
//!   turn, where its EOS ids are unreachable and decode runs to the token limit every
//!   time — see `Tokenizer::encode_chat`. That endpoint would serve the failure mode by
//!   construction.
//! - **Auth, TLS, batching, multi-model.** llama-swap owns model swapping and fronting;
//!   the bind is loopback-only.
//!
//! The context window is fixed at startup (`--ctx`): the KV slabs are allocated once, in
//! `GpuEngine::new`, so a request that will not fit is a 400 rather than a reallocation.

use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::io::BufRead;

pub struct Opts {
    pub port: u16,
    /// KV capacity in tokens, as allocated in `GpuEngine::new`.
    pub ctx: usize,
    /// Reported back as `model` when the request does not name one.
    pub model_id: String,
    /// Speculative decode, already resolved against the artifact by `main`.
    pub mtp: bool,
    pub mtp_min_conf: f32,
    /// `--think`: reason before answering unless the request says otherwise. Off by
    /// default even though the checkpoint's template defaults it ON, because at ~2.7 tok/s
    /// a reasoning block is tens of seconds of silence and most OpenAI clients cannot ask
    /// for it to stop. A request's `enable_thinking` overrides this either way.
    pub think: bool,
}

// ---------------------------------------------------------------------------------------
// The pure half: HTTP framing, message flattening, streaming detok. No engine and no
// backend, so it compiles — and TESTS — in the featureless build that CI runs.
// ---------------------------------------------------------------------------------------

#[derive(Debug, PartialEq)]
struct Req {
    method: String,
    path: String,
    body: Vec<u8>,
}

/// Read one HTTP/1.1 request. `Ok(None)` is a clean EOF — a proxy probing the port.
///
/// Content-Length only: OpenAI clients always send it, and a chunked body arrives here as
/// an empty one and fails as a 400 rather than being half-read into a truncated prompt.
fn read_req<R: BufRead>(r: &mut R) -> Result<Option<Req>> {
    let mut line = String::new();
    if r.read_line(&mut line)? == 0 {
        return Ok(None);
    }
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or_default().to_string();
    // Strip any query string: `/v1/models?limit=1` is the same route.
    let path = parts
        .next()
        .unwrap_or_default()
        .split('?')
        .next()
        .unwrap_or_default()
        .to_string();
    ensure!(
        !method.is_empty() && !path.is_empty(),
        "malformed request line {line:?}"
    );
    let mut len = 0usize;
    loop {
        let mut h = String::new();
        if r.read_line(&mut h)? == 0 {
            break;
        }
        let h = h.trim_end();
        if h.is_empty() {
            break;
        }
        // Header names are case-insensitive and clients disagree about the spelling.
        if let Some((name, v)) = h.split_once(':')
            && name.eq_ignore_ascii_case("content-length")
        {
            len = v.trim().parse().context("Content-Length is not a number")?;
        }
    }
    let mut body = vec![0u8; len];
    std::io::Read::read_exact(r, &mut body).context("short request body")?;
    Ok(Some(Req { method, path, body }))
}

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
fn messages_to_turns(body: &Value) -> Result<Vec<(String, String)>> {
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
                let block = crate::artifact::tokenizer::tool_response_markup(&content);
                match turns.last_mut() {
                    Some((r, c)) if r == "observation" => c.push_str(&block),
                    _ => turns.push(("observation".to_string(), block)),
                }
                continue;
            }
            "assistant" => {
                for (j, tc) in m
                    .get("tool_calls")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .enumerate()
                {
                    let f = tc.get("function").unwrap_or(tc);
                    let name = f.get("name").and_then(Value::as_str).with_context(|| {
                        format!("messages[{i}].tool_calls[{j}] has no function name")
                    })?;
                    // OpenAI sends `arguments` as a JSON *string*; a client that sends an
                    // object instead is accepted rather than argued with.
                    let args = match f.get("arguments") {
                        Some(Value::String(s)) => {
                            serde_json::from_str(s).unwrap_or_else(|_| json!({}))
                        }
                        Some(v) => v.clone(),
                        None => json!({}),
                    };
                    content.push_str(&crate::artifact::tokenizer::tool_call_markup(name, &args));
                }
            }
            _ => {}
        }
        // `developer` is OpenAI's newer name for a system message; the template has no such
        // turn, so it frames as one.
        let role = if role == "developer" { "system" } else { role };
        turns.push((role.to_string(), content));
    }
    Ok(turns)
}

const TOOL_OPEN: &str = "<tool_call>";
const TOOL_CLOSE: &str = "</tool_call>";
const ARG_KEY: (&str, &str) = ("<arg_key>", "</arg_key>");
const ARG_VALUE: (&str, &str) = ("<arg_value>", "</arg_value>");

/// `(inner, rest)` for the first `open`…`close` pair, or `None`.
fn take<'a>(s: &'a str, (open, close): (&str, &str)) -> Option<(&'a str, &'a str)> {
    let i = s.find(open)? + open.len();
    let j = s[i..].find(close)?;
    Some((&s[i..i + j], &s[i + j + close.len()..]))
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
fn parse_tool_calls(text: &str, id: &str) -> (String, Vec<Value>) {
    let (mut prose, mut calls, mut rest) = (String::new(), Vec::new(), text);
    while let Some(i) = rest.find(TOOL_OPEN) {
        prose.push_str(&rest[..i]);
        let after = &rest[i + TOOL_OPEN.len()..];
        let (inner, tail) = match after.find(TOOL_CLOSE) {
            Some(j) => (&after[..j], &after[j + TOOL_CLOSE.len()..]),
            None => (after, ""),
        };
        let name_end = inner.find(ARG_KEY.0).unwrap_or(inner.len());
        let mut args = serde_json::Map::new();
        let mut cursor = &inner[name_end..];
        while let Some((key, r)) = take(cursor, ARG_KEY) {
            let Some((raw, r2)) = take(r, ARG_VALUE) else {
                break;
            };
            let v = serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string()));
            args.insert(key.to_string(), v);
            cursor = r2;
        }
        calls.push(json!({
            "id": format!("call_{id}_{}", calls.len()),
            "type": "function",
            "function": {
                "name": inner[..name_end].trim(),
                // OpenAI's `arguments` is a JSON string, not an object.
                "arguments": Value::Object(args).to_string(),
            }
        }));
        rest = tail;
    }
    prose.push_str(rest);
    (prose.trim().to_string(), calls)
}

/// The part of `text` that is safe to stream as `content` right now.
///
/// Everything before the first `<tool_call>` and not a byte more — a tool call is a protocol
/// message, not prose, and it goes out as a structured delta at the end instead. A trailing
/// PARTIAL marker is held back too: mid-generation the text can end `…<tool_ca`, and emitting
/// that would leak a fragment which the next token turns into a marker, after which `content`
/// would have to shrink. A delta stream cannot express shrinking.
fn streamable(text: &str) -> &str {
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
fn split_think(full: &str, thinking: bool) -> (&str, &str) {
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
fn delta<'a>(sent: &str, full: &'a str) -> Option<&'a str> {
    let stable = full.trim_end_matches('\u{FFFD}');
    // `strip_prefix` rather than slicing at `sent.len()`: if a decode ever fails to extend
    // what we already sent, emitting nothing is wrong but harmless, and a panic is not.
    stable.strip_prefix(sent).filter(|d| !d.is_empty())
}

// ---------------------------------------------------------------------------------------
// The engine half. One `cfg` on the module rather than nine on its items — and the split
// is the same one `lib.rs` draws: everything above this line is the backend-independent
// part, which is the part CI can run.
// ---------------------------------------------------------------------------------------

#[cfg(feature = "rocm")]
pub use live::serve;

#[cfg(feature = "rocm")]
mod live {
    use super::{
        Opts, Req, delta, messages_to_turns, parse_tool_calls, read_req, split_think, streamable,
    };
    use anyhow::{Context, Result, ensure};
    use serde_json::{Value, json};
    use std::io::{BufReader, Write};
    use std::net::TcpStream;
    use std::time::Duration;

    /// Token budget for a request that does not name one. OpenAI's own default is "until
    /// the context runs out", which at this engine's speed is a ~45-minute answer to a
    /// client that only meant to ask a question.
    const DEFAULT_MAX_TOKENS: usize = 512;

    /// How long the accept loop naps between polls while idle. It polls rather than
    /// blocking for exactly one reason, in [`serve`]: the wedge watchdog.
    const IDLE_POLL: Duration = Duration::from_millis(100);

    /// Everything a request handler needs from the process. The three always travel
    /// together — `handle`, `chat` and `parse_ask` had spelled the same list out
    /// separately — and bundling them keeps `&mut GpuEngine` where it belongs: threaded
    /// through, never cloned, exactly one live borrow at a time.
    struct Ctx<'a, 'e> {
        engine: &'a mut crate::gpu::GpuEngine<'e>,
        tok: &'a crate::artifact::tokenizer::Tokenizer,
        opts: &'a Opts,
    }

    /// A client that connects and then says nothing — or stops reading mid-stream — must
    /// not be able to wedge a single-threaded server. It would not just stall this
    /// request: no token would land, and the wedge watchdog would abort the process.
    const CLIENT_TIMEOUT: Duration = Duration::from_secs(30);

    /// Serve until killed. llama-swap owns the process lifetime — it spawns on demand and
    /// kills on TTL — so there is no shutdown endpoint and no graceful drain to write.
    ///
    /// `hb` is the SAME heartbeat the engine beats per token. An idle server generates no
    /// tokens, so without beating it here the wedge watchdog would abort a perfectly
    /// healthy process `RIVOLI_WATCHDOG_SECS` after the last request. That is why the
    /// accept loop polls instead of blocking, and it is the whole reason `IDLE_POLL` exists.
    pub fn serve(
        engine: &mut crate::gpu::GpuEngine<'_>,
        tok: &crate::artifact::tokenizer::Tokenizer,
        hb: &crate::watchdog::Heartbeat,
        opts: &Opts,
    ) -> Result<()> {
        let listener = std::net::TcpListener::bind(("127.0.0.1", opts.port))
            .with_context(|| format!("bind 127.0.0.1:{}", opts.port))?;
        listener.set_nonblocking(true)?;
        // Logged only once the pin and the KV slabs are built, so the port opening IS the
        // readiness signal: llama-swap's health check gets connection-refused until then,
        // which it treats as "not up yet" exactly as it does for llama.cpp. Pin build is
        // ~1 minute, so its `healthCheckTimeout` has to clear that.
        tracing::info!(
            "serving on http://127.0.0.1:{} — POST /v1/chat/completions, GET /v1/models, \
             GET /health | ctx {} tokens, model id {:?}",
            opts.port,
            opts.ctx,
            opts.model_id,
        );
        loop {
            match listener.accept() {
                Ok((sock, _)) => {
                    sock.set_nonblocking(false)?;
                    sock.set_read_timeout(Some(CLIENT_TIMEOUT))?;
                    sock.set_write_timeout(Some(CLIENT_TIMEOUT))?;
                    // Nagle would hold a one-token SSE frame back waiting for company,
                    // which is precisely the latency streaming exists to remove.
                    sock.set_nodelay(true)?;
                    // A per-request failure is the CLIENT's problem, never the server's:
                    // log it and keep the (~1 minute to rebuild) engine alive.
                    let mut cx = Ctx { engine, tok, opts };
                    if let Err(e) = handle(&sock, &mut cx) {
                        tracing::warn!("request failed: {e:#}");
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    hb.beat();
                    std::thread::sleep(IDLE_POLL);
                }
                Err(e) => return Err(e).context("accept"),
            }
        }
    }

    fn handle(sock: &TcpStream, cx: &mut Ctx<'_, '_>) -> Result<()> {
        let mut r = BufReader::new(sock);
        let Some(req) = read_req(&mut r)? else {
            return Ok(()); // bare connect, no request
        };
        let Req { method, path, body } = req;
        let mut w = sock;
        match (method.as_str(), path.as_str()) {
            ("GET", "/health") => send_json(&mut w, 200, &json!({"status": "ok"})),
            ("GET", "/v1/models") => send_json(
                &mut w,
                200,
                &json!({"object": "list", "data": [{
                    "id": cx.opts.model_id, "object": "model",
                    "created": now_secs(), "owned_by": "rivoli",
                }]}),
            ),
            ("POST", "/v1/chat/completions") => chat(&mut w, &body, cx),
            _ => send_json(&mut w, 404, &err_body(&format!("no route {method} {path}"))),
        }
    }

    /// What the request asked for, once it is known to be answerable.
    struct Ask {
        prompt_ids: Vec<u32>,
        ngen: usize,
        stream: bool,
        model: String,
        /// Whether the prompt left `<think>` open, which is the only way to know how to
        /// read the generation back — see `split_think`.
        think: bool,
    }

    fn parse_ask(body: &[u8], cx: &Ctx<'_, '_>) -> Result<Ask> {
        let (tok, opts) = (cx.tok, cx.opts);
        let body: Value = serde_json::from_slice(body).context("body is not JSON")?;
        // `tool_choice` is accepted only in the forms this can honour. "none" is honoured by
        // dropping the declarations, which genuinely prevents calls; "required" is refused
        // rather than faked, because nothing here can force the model's hand and a client
        // that asked for a guaranteed call must not get prose that looks like compliance.
        let choice = body.get("tool_choice");
        match choice.and_then(Value::as_str) {
            None | Some("auto") | Some("none") => {}
            Some(other) => anyhow::bail!(
                "`tool_choice` {other:?} is not supported — this server can do \"auto\" or \
                 \"none\"; it cannot force a call"
            ),
        }
        ensure!(
            !choice.is_some_and(Value::is_object),
            "`tool_choice` naming a specific function is not supported — this server cannot \
             force a call; use \"auto\""
        );
        let tools = body
            .get("tools")
            .filter(|_| choice.and_then(Value::as_str) != Some("none"));
        let turns = messages_to_turns(&body)?;
        // Thinking: the request wins, the server's `--think` is the default, and the
        // checkpoint template's own default (on) is deliberately not inherited — see Opts.
        // `reasoning_effort: "none"` is how OpenAI clients say "don't", so honour that too.
        let effort = body.get("reasoning_effort").and_then(Value::as_str);
        let think = match body.get("enable_thinking").and_then(Value::as_bool) {
            Some(t) => t,
            None => match effort {
                Some("none") => false,
                Some(_) => true,
                None => opts.think,
            },
        };
        let prompt_ids = tok.encode_chat_turns(
            &turns
                .iter()
                .map(|(r, c)| (r.as_str(), c.as_str()))
                .collect::<Vec<_>>(),
            &crate::artifact::tokenizer::ChatOpts {
                thinking: think,
                reasoning_effort: effort,
                tools,
            },
        )?;
        // One slot beyond the prompt for the token the last forward produces; `forward`
        // refuses `pos >= max_ctx`, so this bound is the server's half of that contract.
        let room = opts
            .ctx
            .checked_sub(prompt_ids.len() + 1)
            .filter(|r| *r > 0)
            .with_context(|| {
                format!(
                    "prompt is {} tokens and this server was started with --ctx {}; restart \
                     it with a larger --ctx, or send a shorter conversation",
                    prompt_ids.len(),
                    opts.ctx
                )
            })?;
        let asked = body
            .get("max_tokens")
            .or_else(|| body.get("max_completion_tokens"))
            .and_then(Value::as_u64)
            .map_or(DEFAULT_MAX_TOKENS, |n| n as usize);
        if body.get("temperature").is_some() || body.get("top_p").is_some() {
            warn_sampling_ignored();
        }
        Ok(Ask {
            ngen: asked.min(room),
            stream: body.get("stream").and_then(Value::as_bool).unwrap_or(false),
            model: body
                .get("model")
                .and_then(Value::as_str)
                .unwrap_or(&opts.model_id)
                .to_string(),
            prompt_ids,
            think,
        })
    }

    fn chat(w: &mut impl Write, body: &[u8], cx: &mut Ctx<'_, '_>) -> Result<()> {
        let (tok, opts) = (cx.tok, cx.opts);
        let ask = match parse_ask(body, cx) {
            Ok(a) => a,
            Err(e) => return send_json(w, 400, &err_body(&format!("{e:#}"))),
        };
        let Ask {
            prompt_ids,
            ngen,
            stream,
            model,
            think,
        } = ask;
        let created = now_secs();
        let id = format!("chatcmpl-{created}");
        tracing::info!(
            "chat: {} prompt tokens, up to {ngen} generated, stream={stream}, thinking={think}",
            prompt_ids.len()
        );

        let t0 = std::time::Instant::now();
        let mut hung_up = false;
        // The decode arguments, in ONE place: the two arms below differ only in the
        // per-token callback (stream an SSE delta, or do nothing until the end), and a
        // second copy of the list is where `--mtp-min-conf` would go missing from one arm.
        let mut decode = |on_tok: &mut dyn FnMut(u32) -> bool| {
            cx.engine.generate(
                &prompt_ids,
                ngen,
                &tok.eos,
                opts.mtp,
                opts.mtp_min_conf,
                on_tok,
                // No scripted follow-ups: a request's turns arrive in its `messages` array
                // and are framed by `encode_chat_turns` above. The script is a `-bench`
                // harness only.
                &[],
            )
        };
        let (ids, summary) = if stream {
            sse_head(w)?;
            sse(
                w,
                &chunk(&id, created, &model, json!({"role": "assistant"}), None),
            )?;
            // Decode the whole prefix each token and send what it added. O(n^2) over a
            // generation that arrives at ~3 tok/s, i.e. free — the alternative is an
            // incremental detokenizer, and `delta` documents why that is the fiddly one.
            // ponytail: prefix re-decode; revisit only if generations get long AND fast.
            //
            // Two channels, because `</think>` moves text from one to the other: the token
            // that closes it can legitimately extend the reasoning AND start the content in
            // the same step, so both are checked every time rather than switching once.
            // `reasoning_content` is the field Open WebUI and the OpenAI-compatible
            // ecosystem already read for a collapsible thinking section.
            let mut acc = Vec::with_capacity(ngen);
            let (mut sent_r, mut sent_c, mut live) = (String::new(), String::new(), true);
            let mut on_tok = |t: u32| {
                acc.push(t);
                let Ok(full) = tok.decode_all(&acc) else {
                    return true;
                };
                let (reasoning, content) = split_think(&full, think);
                for (field, sent, target) in [
                    ("reasoning_content", &mut sent_r, reasoning),
                    // Prose only. Tool calls leave as one structured delta once the whole
                    // reply is parseable — streaming their markup would hand the client
                    // `<tool_call>` to render as text.
                    ("content", &mut sent_c, streamable(content)),
                ] {
                    let Some(d) = delta(sent, target) else {
                        continue;
                    };
                    let ev = chunk(&id, created, &model, json!({ field: d }), None);
                    // A write failure IS the client hanging up. Stop the decode: the GPU is
                    // sole tenant, so finishing a generation nobody will read is time stolen
                    // from the next request.
                    if sse(w, &ev).is_err() {
                        live = false;
                        break;
                    }
                    sent.push_str(d);
                }
                live
            };
            let out = decode(&mut on_tok)?;
            hung_up = !live;
            out
        } else {
            decode(&mut |_: u32| true)?
        };

        let text = tok.decode_all(&ids)?;
        let dt = t0.elapsed().as_secs_f64();
        tracing::info!(
            "chat: {} tokens in {dt:.1}s ({:.2} tok/s), expert hit {:.1}%{}",
            ids.len(),
            summary.tok_per_s,
            summary.hit_pct,
            if hung_up { " (client hung up)" } else { "" },
        );
        // The repo's standing rule (README, docs/reference/modes.md): a looped generation is not a slow
        // generation, it is a broken one, and it benchmarks FASTER because it re-routes to
        // the same few experts. Server mode must not be the one path that hides it.
        let rep = crate::telemetry::repetition_report(&text);
        if crate::telemetry::is_degenerate(&rep) {
            tracing::warn!(
                "STRUCTURALLY DEGENERATE response: one line repeats {}x and the distinct-word \
                 ratio is {:.3} (healthy band 0.42-0.53)",
                rep.top_line,
                rep.distinct,
            );
        }
        if stream {
            // The prose already went out chunk by chunk; this is the epilogue. It sits here,
            // after the single `decode_all` above, because the streaming arm used to decode
            // the whole generation a SECOND time to build it.
            if !hung_up {
                // Tool calls go out whole rather than as fragments: the markup is only
                // parseable once closed, and OpenAI's streamed tool-call shape (per-call
                // `index`, arguments assembled across deltas) is a reassembly protocol a
                // client is free to receive in one piece.
                let (_, content) = split_think(&text, think);
                let (_, calls) = parse_tool_calls(content, &id);
                if !calls.is_empty() {
                    let indexed: Vec<Value> = calls
                        .iter()
                        .enumerate()
                        .map(|(i, c)| {
                            let mut c = c.clone();
                            c["index"] = json!(i);
                            c
                        })
                        .collect();
                    sse(
                        w,
                        &chunk(&id, created, &model, json!({ "tool_calls": indexed }), None),
                    )?;
                }
                let reason = stop_reason(&calls, ids.len(), ngen);
                sse(w, &chunk(&id, created, &model, json!({}), Some(reason)))?;
                w.write_all(b"data: [DONE]\n\n")?;
                w.flush()?;
            }
            return Ok(());
        }
        let (reasoning, content) = split_think(&text, think);
        let (prose, calls) = parse_tool_calls(content, &id);
        let mut message = json!({"role": "assistant", "content": prose});
        // Only when there is some: a client that does not know the field should not have to
        // filter an empty one out of every non-thinking response.
        if !reasoning.is_empty() {
            message["reasoning_content"] = json!(reasoning);
        }
        if !calls.is_empty() {
            // OpenAI pairs `tool_calls` with a null content, not an empty string — a client
            // that renders content unconditionally would otherwise print a blank message.
            if prose.is_empty() {
                message["content"] = Value::Null;
            }
            message["tool_calls"] = json!(calls);
        }
        send_json(
            w,
            200,
            &json!({"id": id, "object": "chat.completion", "created": created, "model": model,
                    "choices": [{"index": 0,
                                 "finish_reason": stop_reason(&calls, ids.len(), ngen),
                                 "message": message}],
                    "usage": {"prompt_tokens": prompt_ids.len(), "completion_tokens": ids.len(),
                              "total_tokens": prompt_ids.len() + ids.len()}}),
        )
    }

    /// `tool_calls` outranks `stop` — an agent loop branches on this field, and a reply
    /// carrying calls but reporting `stop` reads as "the model is done talking to you",
    /// which is the opposite of what it means. `length` still wins over both: a call cut
    /// off by the budget may be incomplete, and saying `tool_calls` would assert it is not.
    ///
    /// EOS is the only way a decode ends short of its budget, so reaching it IS `length`.
    fn stop_reason(calls: &[Value], generated: usize, ngen: usize) -> &'static str {
        match (generated >= ngen, calls.is_empty()) {
            (true, _) => "length",
            (false, false) => "tool_calls",
            (false, true) => "stop",
        }
    }

    fn chunk(id: &str, created: u64, model: &str, delta: Value, finish: Option<&str>) -> Value {
        json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": model,
               "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})
    }

    /// Once per process, not once per request: a chat UI sends `temperature` on every turn
    /// and the warning would drown the decode log it shares.
    fn warn_sampling_ignored() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::warn!(
                "`temperature`/`top_p` are IGNORED — this engine decodes greedy argmax, which \
                 is what every number in docs/measurement/benchmarks.md is measured against. Output is \
                 deterministic no matter what the client asks for."
            );
        });
    }

    fn err_body(msg: &str) -> Value {
        json!({"error": {"message": msg, "type": "invalid_request_error"}})
    }

    fn now_secs() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs())
    }

    /// One whole JSON response: status line, `Content-Length`, `Connection: close`, body.
    /// Length-delimited rather than chunked, which is what makes the keep-alive state
    /// machine unnecessary (see the module header).
    fn send_json(w: &mut impl Write, status: u16, body: &Value) -> Result<()> {
        let body = serde_json::to_vec(body)?;
        write!(
            w,
            "HTTP/1.1 {status} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\
             Connection: close\r\n\r\n",
            reason_phrase(status),
            body.len(),
        )?;
        write_flush(w, &body)
    }

    /// Write and flush in one step. Everything this server emits is flushed at the point it
    /// is produced — one request per connection, so there is no later write to piggyback on,
    /// and an unflushed SSE frame is a token that did not stream.
    fn write_flush(w: &mut impl Write, bytes: &[u8]) -> Result<()> {
        w.write_all(bytes)?;
        Ok(w.flush()?)
    }

    /// The SSE preamble — 200 + `text/event-stream`, flushed before the first chunk so the
    /// client can start rendering while the first token is still ~a second away.
    fn sse_head(w: &mut impl Write) -> Result<()> {
        write_flush(
            w,
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n\
              Connection: close\r\n\r\n",
        )
    }

    /// One `data:` event. Returns the write error rather than logging it — the caller
    /// reads a failure here as the client having hung up and stops the decode.
    fn sse(w: &mut impl Write, v: &Value) -> Result<()> {
        write!(w, "data: {}\n\n", serde_json::to_string(v)?)?;
        w.flush()?; // per event: an unflushed token is a token that did not stream
        Ok(())
    }

    fn reason_phrase(status: u16) -> &'static str {
        match status {
            200 => "OK",
            400 => "Bad Request",
            404 => "Not Found",
            _ => "Error",
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::io::BufReader;

    fn req(bytes: &str) -> Option<Req> {
        read_req(&mut BufReader::new(bytes.as_bytes())).unwrap()
    }

    #[test]
    fn reads_a_post_with_a_body() {
        let r = req(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 7\r\n\r\n{\"a\":1}",
        )
        .unwrap();
        assert_eq!(r.method, "POST");
        assert_eq!(r.path, "/v1/chat/completions");
        assert_eq!(r.body, b"{\"a\":1}");
    }

    #[test]
    fn header_case_and_query_string_do_not_change_the_route() {
        let r = req("GET /v1/models?limit=1 HTTP/1.1\r\ncontent-length: 0\r\n\r\n").unwrap();
        assert_eq!(r.path, "/v1/models");
        assert!(r.body.is_empty());
    }

    #[test]
    fn clean_eof_is_not_an_error() {
        assert_eq!(req(""), None);
    }

    #[test]
    fn a_short_body_fails_rather_than_serving_a_truncated_prompt() {
        // Content-Length lies: 99 promised, 7 delivered. Reading what arrived would
        // silently answer a conversation the client did not send.
        assert!(
            read_req(&mut BufReader::new(
                "POST /x HTTP/1.1\r\nContent-Length: 99\r\n\r\n{\"a\":1}".as_bytes()
            ))
            .is_err()
        );
    }

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
}
