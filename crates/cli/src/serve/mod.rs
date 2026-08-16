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
//! **Almost nothing here is model-shaped, and the exception is named.** The engine arrives as
//! `rivoli_engine::Engine`, the one seam, and every architecture-specific decision — which
//! kernels, which KV layout, whether speculative decode exists — is on the far side of it.
//! There is not one `#[cfg]` in this module for the same reason `main.rs` has none: `Engine`
//! is a type in the featureless build too (an uninhabited one), so this whole subtree
//! compiles, lints and runs its tests under a plain `cargo test --workspace`. All still true.
//!
//! > **AMENDED 2026-08-17 (M11b).** This said "Nothing here is model-shaped", full stop, and
//! > that had stopped being true in the worst way: with no architecture here, **every** request
//! > was framed with GLM's chat template and read back with GLM's markers, Muse Glimmer's
//! > included. Framing cannot go behind the seam — a template is a property of the CHECKPOINT,
//! > and `Tokenizer` holds one vocabulary and no opinion about turns — so [`Opts::arch`]
//! > carries it. **Two matches, both exhaustive, both one function**: [`frame_prompt`] on the
//! > way in and [`split_channels`] on the way out. Feeding an instruct model another model's turn
//! > markers puts it outside its turn structure, where its stop ids are unreachable and decode
//! > runs to the limit every time — the failure behind the old tree's 56-run retraction.
//!
//! Split by cohesion, four files:
//! - [`http`] — the HTTP/1.1 and SSE wire format, generic over `BufRead`/`Write`.
//! - [`glimmer`] — Muse Glimmer's request framing and reply channels, split out at M11b
//!   when the two halves pushed `oai` past the 800-line soft cap; the cut is by MODEL, since
//!   `oai` is GLM-shaped end to end.
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

mod glimmer;
mod http;
mod oai;

use anyhow::{Context, Result, ensure};
use rivoli_artifact::tokenizer::{ChatOpts, Tokenizer};
use rivoli_core::legality::Arch;
use rivoli_engine::{Decoded, Engine, GenSpec};
use serde_json::{Value, json};
use std::io::{BufReader, Write};
use std::net::TcpStream;
use std::time::Duration;

pub struct Opts {
    pub port: u16,
    /// Reported back as `model` when the request does not name one.
    pub model_id: String,
    /// `--think`: reason before answering unless the request says otherwise. Why the
    /// default inverts the checkpoint's is `ChatOpts::thinking`'s argument — one home.
    pub think: bool,
    /// **Which chat template frames a request and reads its reply back** — the one
    /// architecture-shaped fact this module carries. See the header's dated amendment.
    ///
    /// An `Arch` rather than a pair of closures: `rivoli_core::legality` already owns the
    /// enum, both matches below are EXHAUSTIVE, and a fifth architecture is then a compile
    /// error rather than whichever arm someone wrote first. That exhaustiveness is the whole
    /// value — an `if arch == MuseGlimmer { … } else { …GLM… }` would have left DeepSeek-V4
    /// silently framed with GLM's markers, which is what this field exists to stop.
    pub arch: Arch,
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
    /// Whether the prompt left `<think>` open, which on GLM's framing is the only way to
    /// know how to read the generation back — see `oai::split_think`. Inert on Glimmer, whose
    /// channels are named in the generated text (`oai::split_glimmer`).
    think: bool,
    /// Which template framed this request, carried so the reply is read back with the SAME
    /// one. On `Ask` rather than threaded through `stream_epilogue`/`json_reply`/the streaming
    /// closure, because it is a property OF THE REQUEST that the reader needs, exactly like
    /// `think` beside it — and because the mismatch this prevents is silent: a Glimmer reply
    /// read with GLM's markers yields an empty `content` or leaks `to=user<|message|>` to the
    /// user's screen.
    arch: Arch,
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
    // One match over the whole value: the string form, the object form (naming a
    // function), and anything else (a number, an array) all land in an arm — the old
    // two-check shape silently ignored the last group (review 2026-08-16).
    let none = match body.get("tool_choice") {
        None => false,
        Some(Value::String(s)) if s == "auto" => false,
        Some(Value::String(s)) if s == "none" => true,
        Some(other) => anyhow::bail!(
            "`tool_choice` {other} is not supported — this server can do \"auto\" or \
             \"none\"; it cannot force a call"
        ),
    };
    Ok(body.get("tools").filter(|_| !none))
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

/// Frame one request in the CHECKPOINT's own chat template.
///
/// **Exhaustive, and the arms differ in SHAPE rather than in a parameter.** GLM flattens
/// `messages` into `(role, content)` turns and builds a token-ID list; Muse Glimmer renders
/// the raw array to a STRING, because its template reads seven optional per-message fields
/// (`tool_calls`, `reasoning_content`, `recipient`, `end_turn`, `name`, `tool_call_id`,
/// content-as-parts) that a flattened turn has already discarded.
///
/// **`messages_to_turns` runs on BOTH arms, and on the Glimmer one only for its refusals** —
/// its flattened output is discarded there. Named because "called for side effects" deserves
/// to be: it refuses a `messages` that is absent, not an array, or empty; a role outside
/// `system|developer|user|assistant|tool`; and a `tool_calls` entry with no `function.name`.
/// The first two are exactly the holes Glimmer's no-`else` role chain leaves open — an
/// unrecognised role renders as NOTHING there. The third is STRICTER than Glimmer's template,
/// which renders a nameless call as `to=`: a 400 naming the message index beats a prompt
/// teaching the model that nameless calls are normal.
///
/// **DeepSeek-V4 and Kimi-K3 take GLM's framing, and that is a KNOWN GAP, not a decision** —
/// the same shape `main.rs::frame_prompt` carried for Glimmer until this milestone closed it.
/// V4 has its own encoder (`tok.encode_dsv4`, which `--bench` already uses) and no multi-turn
/// path from an OpenAI body; K3 ships no template in any tree and `main` refuses `--port` for
/// it before reaching here, so its arm is unreachable and kept only for the exhaustiveness
/// that makes the V4 gap visible. Closing V4's needs its own id-pinned comparison, exactly as
/// this one did.
fn frame_prompt(
    body: &Value,
    tok: &Tokenizer,
    arch: Arch,
    (tools, think, effort): (Option<&Value>, bool, Option<&str>),
) -> Result<Vec<u32>> {
    let turns = oai::messages_to_turns(body)?;
    match arch {
        Arch::MuseGlimmer => tok.encode(&glimmer::glimmer_prompt(
            body,
            (think, effort),
            &rivoli_artifact::glimmer_encoding::utc_date(std::time::SystemTime::now()),
        )?),
        Arch::GlmMoeDsa | Arch::DeepseekV4 | Arch::KimiK3 => tok.encode_chat_turns(
            &turns
                .iter()
                .map(|(r, c)| (r.as_str(), c.as_str()))
                .collect::<Vec<_>>(),
            &ChatOpts {
                thinking: think,
                reasoning_effort: effort,
                tools,
            },
        ),
    }
}

/// Split a generation into `(reasoning, content)` using the CHECKPOINT's own channel marking.
///
/// Feeds [`ReadBack`], which then asks the tool-call question on top.
///
/// The mirror of [`frame_prompt`], and it has to be: a reply framed with one model's markers
/// and read with another's yields either an empty `content` (the split looks for a `</think>`
/// that is not there and calls everything reasoning) or one that leaks raw
/// `to=user<|message|>` into the user's screen. Wiring the request half alone was written and
/// then REVERTED during M11b for exactly that reason; the two halves land together or not
/// at all.
///
/// **`complete` is the parameter the P0 was made of.** `false` comes from `stream_decode`,
/// once per generated token, on a PREFIX; `true` from [`read_back`], once, on the finished
/// text. `oai::split_glimmer`'s doc carries why a prefix must not take the whole-text fallback
/// — it streamed the raw turn header and then wedged the channel for the rest of the request.
///
/// Owned `String`s rather than `&str` slices, because Glimmer's channels are interleaved turns
/// that have to be concatenated while GLM's are one span each. That is an allocation per token
/// in the streaming arm, not per reply — on a path that already re-decodes the whole prefix
/// every token at a few tok/s, which is the measurement that makes it not worth thinking
/// about.
fn split_channels(text: &str, ask: &Ask, complete: bool) -> (String, String) {
    match ask.arch {
        Arch::MuseGlimmer => glimmer::split_glimmer(text, complete),
        Arch::GlmMoeDsa | Arch::DeepseekV4 | Arch::KimiK3 => {
            let (r, c) = oai::split_think(text, ask.think);
            (r.to_string(), c.to_string())
        }
    }
}

fn parse_ask(body: &[u8], cx: &Ctx<'_, '_>) -> Result<Ask> {
    let (tok, opts) = (cx.tok, cx.opts);
    let body: Value = serde_json::from_slice(body).context("body is not JSON")?;
    let tools = tool_declarations(&body)?;
    let (think, effort) = thinking_mode(&body, opts.think);
    let prompt_ids = frame_prompt(&body, tok, opts.arch, (tools, think, effort))?;
    let asked = body
        .get("max_tokens")
        .or_else(|| body.get("max_completion_tokens"))
        .and_then(Value::as_u64)
        .map_or(DEFAULT_MAX_TOKENS, |n| n as usize);
    // The decode loop decides token T before checking the budget, so 0 would come back
    // as one token labelled `finish_reason: "length"` — a reply the client asked not to
    // get. Refused as a 400 instead (review 2026-08-16).
    ensure!(asked >= 1, "max_tokens must be at least 1, got {asked}");
    if body.get("temperature").is_some() || body.get("top_p").is_some() {
        warn_sampling_ignored();
    }
    let created = oai::now_secs();
    Ok(Ask {
        ngen: asked.min(room_for(prompt_ids.len(), cx.engine.max_ctx())?),
        stream: body.get("stream").and_then(Value::as_bool).unwrap_or(false),
        prompt_ids,
        think,
        arch: opts.arch,
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
        let (reasoning, content) = split_channels(&full, ask, false);
        for (field, sent, target) in [
            ("reasoning_content", &mut sent_r, reasoning.as_str()),
            // Prose only. Tool calls leave as one structured delta once the whole
            // reply is parseable — streaming their markup would hand the client
            // `<tool_call>` to render as text.
            ("content", &mut sent_c, oai::streamable(&content)),
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
struct ReadBack {
    /// Owned, not borrowed, since M11b: Glimmer's reasoning is CONCATENATED from however many
    /// `to=self` turns the model emitted, so there is no single slice of the generation to
    /// point at. GLM's one span pays a copy per reply on a path that spends seconds per token.
    reasoning: String,
    prose: String,
    calls: Vec<Value>,
}

fn read_back(text: &str, ask: &Ask) -> ReadBack {
    let (reasoning, content) = split_channels(text, ask, true);
    // **`parse_tool_calls` reads GLM's `<tool_call>` markup and runs on every arm, which is
    // sound only because no arm is TAUGHT another syntax.** `oai::glimmer_prompt` deliberately
    // withholds `tools` from Glimmer's template for exactly this reason — see its doc. On a
    // Glimmer reply this therefore finds nothing and passes the prose through, which is the
    // honest outcome; the day ATEM is advertised, this line is the thing that must change with
    // it.
    let (prose, calls) = oai::parse_tool_calls(&content, &ask.who.id);
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

    /// **The arch DISPATCH, which is the pairing the M11b revert was about.**
    ///
    /// Both leaves are gated in `oai` — `split_glimmer` by six cases plus a prefix-monotonicity
    /// sweep, `split_think` by its own test — and until this existed the `match` that pairs a
    /// request's framing with its reply's reader was not. That is precisely the seam that
    /// failed: the reverted draft framed Glimmer requests with Glimmer's template and read the
    /// replies back with GLM's, and every leaf test stayed green while doing it.
    ///
    /// One text, both arms, opposite answers. Glimmer's markers are inert to `split_think`
    /// (it hunts `</think>`), so a mis-wired dispatch shows up as the raw turn header reaching
    /// `content` — exactly what a user would have seen.
    #[test]
    fn the_reply_is_read_back_with_the_same_template_that_framed_it() {
        let ask = |arch| Ask {
            prompt_ids: vec![],
            ngen: 1,
            stream: false,
            think: true,
            arch,
            who: oai::Completion {
                id: String::new(),
                created: 0,
                model: String::new(),
            },
        };
        let reply = " to=user<|message|>hi<|eot|>";
        assert_eq!(
            split_channels(reply, &ask(Arch::MuseGlimmer), true),
            (String::new(), "hi".to_string()),
            "the Glimmer arm must strip the turn markers"
        );
        // GLM's reader is told `thinking: true` and finds no `</think>`, so it calls the whole
        // thing reasoning and leaves content EMPTY — the visible signature of a mis-wired
        // dispatch, and the reason this pairing needs its own gate.
        assert_eq!(
            split_channels(reply, &ask(Arch::GlmMoeDsa), true),
            (reply.to_string(), String::new()),
            "the GLM arm must be left alone; if this ever equals the Glimmer answer, the \
             dispatch has collapsed and nothing else here would notice"
        );
    }

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
