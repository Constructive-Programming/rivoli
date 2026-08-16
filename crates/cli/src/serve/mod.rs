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
//! **Nothing here is model-shaped.** The engine arrives as `rivoli_engine::Engine`, the one
//! seam, and every architecture-specific decision — which kernels, which KV layout, whether
//! speculative decode exists — is on the far side of it. There is not one `#[cfg]` in this
//! module for the same reason `main.rs` has none: `Engine` is a type in the featureless
//! build too (an uninhabited one), so this whole subtree compiles, lints and runs its tests
//! under a plain `cargo test --workspace`.
//!
//! Split by cohesion, three files:
//! - [`http`] — the HTTP/1.1 and SSE wire format, generic over `BufRead`/`Write`.
//! - [`oai`] — the OpenAI chat-completion semantics: `messages` → template turns, tool-call
//!   parsing, think splitting, streaming deltas. Pure functions, and where the tests are.
//! - this file — the socket loop and one request's lifecycle, the only part that needs an
//!   engine and therefore the only part a host test cannot reach.
//!
//! Deliberately absent, so nobody goes looking:
//! - **Sampling.** The engine is greedy argmax and every number in `docs/measurement/`
//!   is measured that way. `temperature`/`top_p` are accepted and IGNORED, with one warning
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
//! `Engine::open`, so a request that will not fit is a 400 rather than a reallocation. The
//! ceiling is read back from the engine (`Engine::max_ctx`) rather than carried here a
//! second time — see that method for why a copy would be free to drift.

mod http;
mod oai;

use anyhow::{Context, Result, ensure};
use rivoli_artifact::tokenizer::{ChatOpts, Tokenizer};
use rivoli_engine::{Decoded, Engine, GenSpec};
use serde_json::{Value, json};
use std::io::{BufReader, Write};
use std::net::TcpStream;
use std::time::Duration;

pub struct Opts {
    pub port: u16,
    /// Reported back as `model` when the request does not name one.
    pub model_id: String,
    /// `--think`: reason before answering unless the request says otherwise. Off by
    /// default even though the checkpoint's template defaults it ON, because at ~2.7 tok/s
    /// a reasoning block is tens of seconds of silence and most OpenAI clients cannot ask
    /// for it to stop. A request's `enable_thinking` overrides this either way.
    pub think: bool,
}

// NOTE on what `Opts` does NOT carry. The reference server passed `mtp`/`mtp_min_conf`
// through to every decode; here `GenSpec` has no speculative knob and
// `rivoli_core::legality` refuses `--mtp` on the only architecture with a decode path. A
// field kept against the day it returns would be a second authority on whether speculative
// decode exists, and the seam is the first.

/// Token budget for a request that does not name one. OpenAI's own default is "until
/// the context runs out", which at this engine's speed is a ~45-minute answer to a
/// client that only meant to ask a question.
const DEFAULT_MAX_TOKENS: usize = 512;

/// A client that connects and then says nothing — or stops reading mid-stream — must
/// not be able to wedge a single-threaded server.
const CLIENT_TIMEOUT: Duration = Duration::from_secs(30);

/// Everything a request handler needs from the process. The three always travel
/// together — `handle`, `chat` and `parse_ask` had spelled the same list out
/// separately — and bundling them keeps `&mut Engine` where it belongs: threaded
/// through, never cloned, exactly one live borrow at a time.
struct Ctx<'a, 'e> {
    engine: &'a mut Engine<'e>,
    tok: &'a Tokenizer,
    opts: &'a Opts,
}

/// Serve until killed. llama-swap owns the process lifetime — it spawns on demand and
/// kills on TTL — so there is no shutdown endpoint and no graceful drain to write.
///
/// The accept is BLOCKING. The reference server polled with a 100 ms nap for exactly one
/// reason: it beat a wedge watchdog from the idle loop, since an idle server generates no
/// tokens and would otherwise be aborted as hung. This tree has no watchdog, so the poll
/// would be a spin with nothing to feed — if a watchdog lands, the poll comes back with
/// it, and not before.
pub fn serve(engine: &mut Engine<'_>, tok: &Tokenizer, opts: &Opts) -> Result<()> {
    let listener = std::net::TcpListener::bind(("127.0.0.1", opts.port))
        .with_context(|| format!("bind 127.0.0.1:{}", opts.port))?;
    // Logged only once the pin and the KV slabs are built, so the port opening IS the
    // readiness signal: llama-swap's health check gets connection-refused until then,
    // which it treats as "not up yet" exactly as it does for llama.cpp. Pin build is
    // ~1 minute, so its `healthCheckTimeout` has to clear that.
    tracing::info!(
        "serving on http://127.0.0.1:{} — POST /v1/chat/completions, GET /v1/models, \
         GET /health | ctx {} tokens, model id {:?}",
        opts.port,
        engine.max_ctx(),
        opts.model_id,
    );
    loop {
        let (sock, _) = listener.accept().context("accept")?;
        sock.set_read_timeout(Some(CLIENT_TIMEOUT))?;
        sock.set_write_timeout(Some(CLIENT_TIMEOUT))?;
        // Nagle would hold a one-token SSE frame back waiting for company, which is
        // precisely the latency streaming exists to remove.
        sock.set_nodelay(true)?;
        // A per-request failure is the CLIENT's problem, never the server's: log it and
        // keep the (~1 minute to rebuild) engine alive.
        let mut cx = Ctx { engine, tok, opts };
        if let Err(e) = handle(&sock, &mut cx) {
            tracing::warn!("request failed: {e:#}");
        }
    }
}

fn handle(sock: &TcpStream, cx: &mut Ctx<'_, '_>) -> Result<()> {
    let mut r = BufReader::new(sock);
    let Some(req) = http::read_req(&mut r)? else {
        return Ok(()); // bare connect, no request
    };
    let mut w = sock;
    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/health") => http::send_json(&mut w, 200, &json!({"status": "ok"})),
        ("GET", "/v1/models") => http::send_json(
            &mut w,
            200,
            &json!({"object": "list", "data": [{
                "id": cx.opts.model_id, "object": "model",
                "created": oai::now_secs(), "owned_by": "rivoli",
            }]}),
        ),
        ("POST", "/v1/chat/completions") => chat(&mut w, &req.body, cx),
        _ => {
            let msg = format!("no route {} {}", req.method, req.path);
            http::send_json(&mut w, 404, &oai::err_body(&msg))
        }
    }
}

/// What the request asked for, once it is known to be answerable, plus the identity the
/// reply carries. One bundle rather than a parameter list threaded through both arms.
struct Ask {
    prompt_ids: Vec<u32>,
    ngen: usize,
    stream: bool,
    /// Whether the prompt left `<think>` open, which is the only way to know how to
    /// read the generation back — see `oai::split_think`.
    think: bool,
    /// Who is answering — see [`oai::Completion`] for why the three fields travel together.
    who: oai::Completion,
}

impl Ask {
    /// The decode arguments, in ONE place: the two arms below differ only in the per-token
    /// callback (stream an SSE delta, or do nothing until the end), and a second copy of
    /// the list is where a future knob would go missing from one arm.
    fn spec<'s>(&'s self, tok: &'s Tokenizer) -> GenSpec<'s> {
        GenSpec {
            prompt: &self.prompt_ids,
            ngen: self.ngen,
            eos: &tok.eos,
        }
    }
}

/// `tools`, as the request's `tool_choice` leaves them — `None` when the client asked for
/// no calls at all.
///
/// `tool_choice` is accepted only in the forms this can honour. `"none"` is honoured by
/// dropping the declarations, which genuinely prevents calls; `"required"` is refused
/// rather than faked, because nothing here can force the model's hand and a client that
/// asked for a guaranteed call must not get prose that looks like compliance.
fn tool_declarations(body: &Value) -> Result<Option<&Value>> {
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
    Ok(body
        .get("tools")
        .filter(|_| choice.and_then(Value::as_str) != Some("none")))
}

/// Whether to reason first, and at what stated effort.
///
/// The request wins, the server's `--think` is the default, and the checkpoint template's
/// own default (on) is deliberately not inherited — see [`Opts`]. `reasoning_effort:
/// "none"` is how OpenAI clients say "don't", so honour that too.
fn thinking_mode(body: &Value, default: bool) -> (bool, Option<&str>) {
    let effort = body.get("reasoning_effort").and_then(Value::as_str);
    let think = match body.get("enable_thinking").and_then(Value::as_bool) {
        Some(t) => t,
        None => match effort {
            Some("none") => false,
            Some(_) => true,
            None => default,
        },
    };
    (think, effort)
}

fn parse_ask(body: &[u8], cx: &Ctx<'_, '_>) -> Result<Ask> {
    let (tok, opts) = (cx.tok, cx.opts);
    let body: Value = serde_json::from_slice(body).context("body is not JSON")?;
    let tools = tool_declarations(&body)?;
    let turns = oai::messages_to_turns(&body)?;
    let (think, effort) = thinking_mode(&body, opts.think);
    let prompt_ids = tok.encode_chat_turns(
        &turns
            .iter()
            .map(|(r, c)| (r.as_str(), c.as_str()))
            .collect::<Vec<_>>(),
        &ChatOpts {
            thinking: think,
            reasoning_effort: effort,
            tools,
        },
    )?;
    let asked = body
        .get("max_tokens")
        .or_else(|| body.get("max_completion_tokens"))
        .and_then(Value::as_u64)
        .map_or(DEFAULT_MAX_TOKENS, |n| n as usize);
    if body.get("temperature").is_some() || body.get("top_p").is_some() {
        warn_sampling_ignored();
    }
    let created = oai::now_secs();
    Ok(Ask {
        ngen: asked.min(room_for(prompt_ids.len(), cx.engine.max_ctx())?),
        stream: body.get("stream").and_then(Value::as_bool).unwrap_or(false),
        prompt_ids,
        think,
        who: oai::Completion {
            id: format!("chatcmpl-{created}"),
            created,
            model: body
                .get("model")
                .and_then(Value::as_str)
                .unwrap_or(&opts.model_id)
                .to_string(),
        },
    })
}

/// How many tokens are left to generate after a prompt of `prompt` tokens.
///
/// One slot beyond the prompt for the token the last forward produces; `forward` refuses
/// `pos >= max_ctx`, so this bound is the server's half of that contract. A conversation
/// that does not fit is refused here — the KV slabs were allocated once, at startup, and
/// there is nothing to grow.
fn room_for(prompt: usize, max_ctx: usize) -> Result<usize> {
    max_ctx
        .checked_sub(prompt + 1)
        .filter(|r| *r > 0)
        .with_context(|| {
            format!(
                "prompt is {prompt} tokens and this server was started with --ctx \
                 {max_ctx}; restart it with a larger --ctx, or send a shorter conversation"
            )
        })
}

fn chat(w: &mut impl Write, body: &[u8], cx: &mut Ctx<'_, '_>) -> Result<()> {
    let tok = cx.tok;
    let ask = match parse_ask(body, cx) {
        Ok(a) => a,
        Err(e) => return http::send_json(w, 400, &oai::err_body(&format!("{e:#}"))),
    };
    tracing::info!(
        "chat: {} prompt tokens, up to {} generated, stream={}, thinking={}",
        ask.prompt_ids.len(),
        ask.ngen,
        ask.stream,
        ask.think
    );
    let t0 = std::time::Instant::now();
    let (out, hung_up) = match ask.stream {
        true => stream_decode(w, cx, &ask)?,
        false => (cx.engine.generate(ask.spec(tok), &mut |_| true)?, false),
    };
    let text = tok.decode_all(&out.ids)?;
    report(&out, &text, t0.elapsed().as_secs_f64(), hung_up);
    match (ask.stream, hung_up) {
        // Nothing more to say to a socket that stopped reading, and the prose already went
        // out chunk by chunk.
        (true, true) => Ok(()),
        (true, false) => stream_epilogue(w, &ask, &text, out.ids.len()),
        (false, _) => json_reply(w, &ask, &text, out.ids.len()),
    }
}

/// Decode with an SSE delta per token, returning the generation and whether the client hung
/// up mid-stream.
fn stream_decode(w: &mut impl Write, cx: &mut Ctx<'_, '_>, ask: &Ask) -> Result<(Decoded, bool)> {
    let tok = cx.tok;
    http::sse_head(w)?;
    http::sse(w, &ask.who.chunk(json!({"role": "assistant"}), None))?;
    // Decode the whole prefix each token and send what it added. O(n^2) over a
    // generation that arrives at ~3 tok/s, i.e. free — the alternative is an
    // incremental detokenizer, and `oai::delta` documents why that is the fiddly one.
    // ponytail: prefix re-decode; revisit only if generations get long AND fast.
    //
    // Two channels, because `</think>` moves text from one to the other: the token
    // that closes it can legitimately extend the reasoning AND start the content in
    // the same step, so both are checked every time rather than switching once.
    // `reasoning_content` is the field Open WebUI and the OpenAI-compatible
    // ecosystem already read for a collapsible thinking section.
    let mut acc = Vec::with_capacity(ask.ngen);
    let (mut sent_r, mut sent_c, mut live) = (String::new(), String::new(), true);
    let mut on_tok = |t: u32| {
        acc.push(t);
        let Ok(full) = tok.decode_all(&acc) else {
            return true;
        };
        let (reasoning, content) = oai::split_think(&full, ask.think);
        for (field, sent, target) in [
            ("reasoning_content", &mut sent_r, reasoning),
            // Prose only. Tool calls leave as one structured delta once the whole
            // reply is parseable — streaming their markup would hand the client
            // `<tool_call>` to render as text.
            ("content", &mut sent_c, oai::streamable(content)),
        ] {
            let Some(d) = oai::delta(sent, target) else {
                continue;
            };
            // A write failure IS the client hanging up. Stop the decode: the GPU is
            // sole tenant, so finishing a generation nobody will read is time stolen
            // from the next request.
            if http::sse(w, &ask.who.chunk(json!({ field: d }), None)).is_err() {
                live = false;
                break;
            }
            sent.push_str(d);
        }
        live
    };
    let out = cx.engine.generate(ask.spec(tok), &mut on_tok)?;
    Ok((out, !live))
}

/// The generation, read back the way the protocol reports it.
///
/// ONE reader for both arms: the streaming epilogue and the JSON body ask the same three
/// questions of the same text, and answering them twice is how the two would come to
/// disagree about where the reasoning ended.
struct ReadBack<'t> {
    reasoning: &'t str,
    prose: String,
    calls: Vec<Value>,
}

fn read_back<'t>(text: &'t str, ask: &Ask) -> ReadBack<'t> {
    let (reasoning, content) = oai::split_think(text, ask.think);
    let (prose, calls) = oai::parse_tool_calls(content, &ask.who.id);
    ReadBack {
        reasoning,
        prose,
        calls,
    }
}

/// What a finished stream still owes the client: the tool calls, the stop reason, `[DONE]`.
///
/// It sits here, after the single `decode_all` in [`chat`], because the streaming arm used
/// to decode the whole generation a SECOND time to build it.
fn stream_epilogue(w: &mut impl Write, ask: &Ask, text: &str, generated: usize) -> Result<()> {
    let rb = read_back(text, ask);
    // Tool calls go out whole rather than as fragments: the markup is only parseable once
    // closed, and OpenAI's streamed tool-call shape (per-call `index`, arguments assembled
    // across deltas) is a reassembly protocol a client is free to receive in one piece.
    if !rb.calls.is_empty() {
        let indexed: Vec<Value> = rb
            .calls
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let mut c = c.clone();
                c["index"] = json!(i);
                c
            })
            .collect();
        http::sse(w, &ask.who.chunk(json!({ "tool_calls": indexed }), None))?;
    }
    let reason = oai::stop_reason(&rb.calls, generated, ask.ngen);
    http::sse(w, &ask.who.chunk(json!({}), Some(reason)))?;
    http::sse_done(w)
}

fn json_reply(w: &mut impl Write, ask: &Ask, text: &str, generated: usize) -> Result<()> {
    let rb = read_back(text, ask);
    let mut message = json!({"role": "assistant", "content": rb.prose});
    // Only when there is some: a client that does not know the field should not have to
    // filter an empty one out of every non-thinking response.
    if !rb.reasoning.is_empty() {
        message["reasoning_content"] = json!(rb.reasoning);
    }
    if !rb.calls.is_empty() {
        // OpenAI pairs `tool_calls` with a null content, not an empty string — a client
        // that renders content unconditionally would otherwise print a blank message.
        if rb.prose.is_empty() {
            message["content"] = Value::Null;
        }
        message["tool_calls"] = json!(rb.calls);
    }
    let prompt = ask.prompt_ids.len();
    http::send_json(
        w,
        200,
        &json!({"id": ask.who.id, "object": "chat.completion", "created": ask.who.created,
                "model": ask.who.model,
                "choices": [{"index": 0,
                             "finish_reason": oai::stop_reason(&rb.calls, generated, ask.ngen),
                             "message": message}],
                "usage": {"prompt_tokens": prompt, "completion_tokens": generated,
                          "total_tokens": prompt + generated}}),
    )
}

/// The per-request log line, plus the one quality check server mode must not be allowed to
/// skip.
///
/// The repo's standing rule: a looped generation is not a slow generation, it is a broken
/// one, and it benchmarks FASTER because it re-routes to the same few experts. Server mode
/// must not be the one path that hides it.
fn report(out: &Decoded, text: &str, wall_s: f64, hung_up: bool) {
    tracing::info!(
        "chat: {} tokens in {wall_s:.1}s ({:.2} tok/s), {} expert hits, {} misses{}",
        out.ids.len(),
        out.stats.tok_s,
        out.stats.hits,
        out.stats.misses,
        if hung_up { " (client hung up)" } else { "" },
    );
    let rep = rivoli_engine::telemetry::degeneracy::repetition_report(text);
    if rivoli_engine::telemetry::degeneracy::is_degenerate(&rep) {
        tracing::warn!(
            "STRUCTURALLY DEGENERATE response: one line repeats {}x and the distinct-word \
             ratio is {:.3} (healthy band 0.42-0.53)",
            rep.top_line,
            rep.distinct,
        );
    }
}

/// Once per process, not once per request: a chat UI sends `temperature` on every turn
/// and the warning would drown the decode log it shares.
fn warn_sampling_ignored() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        tracing::warn!(
            "`temperature`/`top_p` are IGNORED — this engine decodes greedy argmax, which \
             is what every published number is measured against. Output is deterministic \
             no matter what the client asks for."
        );
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The ceiling contract, stated on the arithmetic rather than on a socket: one slot is
    /// reserved beyond the prompt for the token the last forward produces, and a prompt
    /// that leaves no room is an error (which `chat` turns into a 400) rather than a
    /// silently truncated conversation.
    #[test]
    fn room_leaves_one_slot_for_the_last_forward() {
        assert_eq!(room_for(10, 100).ok(), Some(89));
        // Exactly one slot short of the ceiling: no room to generate anything.
        assert!(room_for(99, 100).is_err());
        assert!(room_for(100, 100).is_err());
        assert!(room_for(4096, 100).is_err());
    }

    /// The refusal names both numbers, because "restart with a larger --ctx" is only
    /// actionable if the operator can see by how much.
    #[test]
    fn an_over_long_prompt_says_what_it_was_and_what_the_ceiling_is() {
        let e = match room_for(9000, 4096) {
            Ok(r) => panic!("9000 tokens fit in 4096? room {r}"),
            Err(e) => e.to_string(),
        };
        assert!(e.contains("9000") && e.contains("4096"), "{e}");
    }

    #[test]
    fn tool_choice_none_drops_the_declarations_and_required_is_refused() {
        let with_tools = json!({"tools": [{"type": "function"}]});
        assert!(tool_declarations(&with_tools).ok().flatten().is_some());
        let none = json!({"tools": [{"type": "function"}], "tool_choice": "none"});
        assert!(tool_declarations(&none).ok().flatten().is_none());
        // Refused, not faked: nothing here can force the model's hand.
        assert!(tool_declarations(&json!({"tool_choice": "required"})).is_err());
        assert!(tool_declarations(&json!({"tool_choice": {"function": {"name": "wx"}}})).is_err());
    }

    /// The request outranks `--think` in BOTH directions, and `reasoning_effort` is the
    /// fallback channel for clients that do not know the vendor field.
    #[test]
    fn the_request_overrides_the_servers_think_default() {
        assert!(thinking_mode(&json!({}), true).0);
        assert!(!thinking_mode(&json!({}), false).0);
        assert!(!thinking_mode(&json!({"enable_thinking": false}), true).0);
        assert!(thinking_mode(&json!({"enable_thinking": true}), false).0);
        // `reasoning_effort` only decides when `enable_thinking` is absent.
        assert!(!thinking_mode(&json!({"reasoning_effort": "none"}), true).0);
        assert_eq!(
            thinking_mode(&json!({"reasoning_effort": "high"}), false),
            (true, Some("high"))
        );
        assert!(
            !thinking_mode(
                &json!({"enable_thinking": false, "reasoning_effort": "high"}),
                true
            )
            .0
        );
    }
}
