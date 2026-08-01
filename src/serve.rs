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
//! is why thinking is exposed as the protocol fields those clients already send and read
//! (`enable_thinking`, `reasoning_content`) rather than as anything rendered, and why tool
//! calling is refused outright below instead of half-built.
//!
//! Deliberately absent, so nobody goes looking:
//! - **Sampling.** The engine is greedy argmax and every number in `benchmarks.md` is
//!   measured that way. `temperature`/`top_p` are accepted and IGNORED, with one warning
//!   per process — honouring them is not a server-side change, and dropping them silently
//!   would leave a client believing its own determinism story.
//! - **Tool calling.** The checkpoint's template does define it (`<tool_call>` /
//!   `<arg_key>` / `<|observation|>`), so this is a real omission rather than an
//!   impossibility — but a request carrying `tools` is refused with a 400 instead of being
//!   answered in prose, because a client that asked for a tool call and got an essay waits
//!   forever. Orchestration lives in the agent above us.
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
use serde_json::Value;
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

/// The roles the hand-ported template can frame. `tool` is absent on purpose: it only
/// appears in a conversation that used `tools`, which `parse_ask` refuses, and framing it
/// as a user turn would feed the model a tool result it has no protocol for.
const ROLES: [&str; 4] = ["system", "developer", "user", "assistant"];

fn messages_to_turns(body: &Value) -> Result<Vec<(String, String)>> {
    let msgs = body
        .get("messages")
        .and_then(Value::as_array)
        .context("`messages` must be an array")?;
    ensure!(!msgs.is_empty(), "`messages` is empty");
    msgs.iter()
        .enumerate()
        .map(|(i, m)| {
            let role = m.get("role").and_then(Value::as_str).unwrap_or("user");
            ensure!(
                ROLES.contains(&role),
                "messages[{i}].role is {role:?}; this server frames {ROLES:?} only\
                 {}",
                if role == "tool" {
                    " — `tools` is not implemented, so a tool result cannot be answered"
                } else {
                    ""
                }
            );
            // `developer` is OpenAI's newer name for a system message; the template has no
            // such turn, so it frames as one.
            let role = if role == "developer" { "system" } else { role };
            Ok((role.to_string(), content_text(m.get("content"))))
        })
        .collect()
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

#[cfg(any(feature = "rocm", feature = "vulkan"))]
pub use live::serve;

#[cfg(any(feature = "rocm", feature = "vulkan"))]
mod live {
    use super::{Opts, Req, delta, messages_to_turns, read_req, split_think};
    use anyhow::{Context, Result};
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
                    if let Err(e) = handle(&sock, engine, tok, opts) {
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

    fn handle(
        sock: &TcpStream,
        engine: &mut crate::gpu::GpuEngine<'_>,
        tok: &crate::artifact::tokenizer::Tokenizer,
        opts: &Opts,
    ) -> Result<()> {
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
                    "id": opts.model_id, "object": "model",
                    "created": now_secs(), "owned_by": "rivoli",
                }]}),
            ),
            ("POST", "/v1/chat/completions") => chat(&mut w, &body, engine, tok, opts),
            _ => send_json(
                &mut w,
                404,
                &err_body(&format!("no route {method} {path}")),
            ),
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

    fn parse_ask(
        body: &[u8],
        tok: &crate::artifact::tokenizer::Tokenizer,
        opts: &Opts,
    ) -> Result<Ask> {
        let body: Value = serde_json::from_slice(body).context("body is not JSON")?;
        // Refused, not ignored. The model CAN call tools — the template defines the whole
        // `<tool_call><arg_key>` syntax — but nothing here renders the declarations or
        // parses the calls back, so answering in prose would leave a client blocked on a
        // tool call that is never coming. See the module doc.
        if body
            .get("tools")
            .and_then(Value::as_array)
            .is_some_and(|t| !t.is_empty())
        {
            anyhow::bail!(
                "`tools` is not implemented by this server. The checkpoint supports tool \
                 calling, but rendering and parsing it is not built here — run the \
                 orchestration in the agent above, or send the request without `tools`"
            );
        }
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

    fn chat(
        w: &mut impl Write,
        body: &[u8],
        engine: &mut crate::gpu::GpuEngine<'_>,
        tok: &crate::artifact::tokenizer::Tokenizer,
        opts: &Opts,
    ) -> Result<()> {
        let ask = match parse_ask(body, tok, opts) {
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
        let (ids, summary) = if stream {
            sse_head(w)?;
            sse(w, &chunk(&id, created, &model, json!({"role": "assistant"}), None))?;
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
                    ("content", &mut sent_c, content),
                ] {
                    let Some(d) = delta(sent, target) else { continue };
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
            let out = engine.generate(
                &prompt_ids,
                ngen,
                &tok.eos,
                opts.mtp,
                opts.mtp_min_conf,
                &mut on_tok,
            )?;
            hung_up = !live;
            if !hung_up {
                let reason = finish_reason(out.0.len(), ngen);
                sse(w, &chunk(&id, created, &model, json!({}), Some(reason)))?;
                w.write_all(b"data: [DONE]\n\n")?;
                w.flush()?;
            }
            out
        } else {
            engine.generate(
                &prompt_ids,
                ngen,
                &tok.eos,
                opts.mtp,
                opts.mtp_min_conf,
                &mut |_: u32| true,
            )?
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
        // The repo's standing rule (README, MODES.md): a looped generation is not a slow
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
            return Ok(()); // already sent, chunk by chunk
        }
        let (reasoning, content) = split_think(&text, think);
        let mut message = json!({"role": "assistant", "content": content});
        // Only when there is some: a client that does not know the field should not have to
        // filter an empty one out of every non-thinking response.
        if !reasoning.is_empty() {
            message["reasoning_content"] = json!(reasoning);
        }
        send_json(
            w,
            200,
            &json!({"id": id, "object": "chat.completion", "created": created, "model": model,
                    "choices": [{"index": 0, "finish_reason": finish_reason(ids.len(), ngen),
                                 "message": message}],
                    "usage": {"prompt_tokens": prompt_ids.len(), "completion_tokens": ids.len(),
                              "total_tokens": prompt_ids.len() + ids.len()}}),
        )
    }

    /// EOS is the only way a decode ends short of its budget, so hitting the budget is
    /// exactly `length`.
    fn finish_reason(generated: usize, ngen: usize) -> &'static str {
        if generated >= ngen { "length" } else { "stop" }
    }

    fn chunk(
        id: &str,
        created: u64,
        model: &str,
        delta: Value,
        finish: Option<&str>,
    ) -> Value {
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
                 is what every number in benchmarks.md is measured against. Output is \
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

    fn send_json(w: &mut impl Write, status: u16, body: &Value) -> Result<()> {
        let body = serde_json::to_vec(body)?;
        write!(
            w,
            "HTTP/1.1 {status} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\
             Connection: close\r\n\r\n",
            reason_phrase(status),
            body.len(),
        )?;
        w.write_all(&body)?;
        w.flush()?;
        Ok(())
    }

    fn sse_head(w: &mut impl Write) -> Result<()> {
        w.write_all(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n\
              Connection: close\r\n\r\n",
        )?;
        w.flush()?;
        Ok(())
    }

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
    fn developer_is_a_system_turn_and_tool_is_refused() {
        // OpenAI renamed `system` to `developer`; the template has only the one turn token.
        assert_eq!(
            messages_to_turns(&json!({"messages": [{"role": "developer", "content": "x"}]}))
                .unwrap(),
            vec![("system".to_string(), "x".to_string())]
        );
        // A tool RESULT cannot be framed when tool CALLS are refused — answering it as a
        // user turn would feed the model a protocol it was not given.
        let e = messages_to_turns(&json!({"messages": [{"role": "tool", "content": "42"}]}))
            .unwrap_err()
            .to_string();
        assert!(e.contains("tools"), "{e}");
    }

    #[test]
    fn split_think_needs_to_be_told_which_mode_it_is_in() {
        // Thinking off: the prompt already closed <think>, so nothing is reasoning. Guessing
        // from the text would make a whole answer disappear into the reasoning channel.
        assert_eq!(split_think("the sky is blue", false), ("", "the sky is blue"));
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
