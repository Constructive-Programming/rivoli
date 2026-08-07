---
scope: glm
status: live
verdict: The OpenAI HTTP server (--port). Thinking defaults OFF and is a prompt prefill, not a flag; tool calling works; sampling and /v1/completions do not, on purpose.
---

# Serving — the OpenAI API, thinking, and tools

`rivoli <artifact> --port 8080 --ctx 8192` serves `POST /v1/chat/completions` (streaming and
not), `GET /v1/models` and `GET /health` on loopback. This is how llama-swap, Open WebUI and
the Hermes agent reach the engine.

**It is an inference backend, not a chat product.** The conversation surface lives above it.
Everything here is protocol: fields those clients already send and read, nothing rendered,
nothing orchestrated, no loop.

## llama-swap

```yaml
models:
  glm-5.2-rivoli:
    cmd: /path/to/rivoli /var/db/rivoli/glm52-vq3-full --port ${PORT} --ctx 8192 --max-mem 115
    proxy: http://127.0.0.1:${PORT}
    checkEndpoint: /health
    healthCheckTimeout: 300
```

`healthCheckTimeout` has to clear the pin build (~1–2 min). The port opens **only once the
model is loaded**, so a health check gets connection-refused rather than a 503 until the
engine is genuinely ready — which is exactly the readiness signal llama-swap wants.

One request at a time, `Connection: close`, no keep-alive state machine. That is a
consequence of the engine rather than a shortcut: the GPU is sole-tenant and decode is ~2.7
tok/s, so a connection pool could only queue what the device already serialises.

> **The wedge watchdog and an idle server are in tension, and it is not obvious.**
> `watchdog::spawn` aborts the process when no token lands for `RIVOLI_WATCHDOG_SECS`
> (default 60). A server is idle by definition, so `serve` polls a non-blocking `accept` and
> beats the same heartbeat. A blocking `accept` would let the wedge detector kill a perfectly
> healthy server a minute after startup, and it would look like a GPU hang.

## Thinking

The checkpoint is a **thinking model**, and thinking is a *prefill*, not a flag the model
reads. The prompt ends at an open `<think>` and the model reasons until it emits `</think>`;
ending it at `<think></think>` instead means it answers straight away. The vocabulary has a
`/nothink` token (154851) and **the template never emits it** — reaching for it is the wrong
instinct.

**rivoli defaults thinking OFF, which is the opposite of the checkpoint's template.** At ~2.7
tok/s a reasoning block is tens of seconds of silence before the first word, and most OpenAI
clients cannot ask for it to stop once it starts. Turn it on per server with `--think`, or
per request:

```jsonc
{"messages": [...], "enable_thinking": true}      // explicit; wins over --think
{"messages": [...], "reasoning_effort": "high"}   // implies thinking. "none" implies off
```

Reasoning comes back in **`reasoning_content`**, never mixed into `content` — the field Open
WebUI renders as a collapsible section. Streaming sends `reasoning_content` deltas and then
`content` deltas; the token that closes `</think>` can extend one and start the other in the
same step, so both channels are checked every token rather than switching once.

A generation that exhausts its budget mid-reasoning returns reasoning and an **empty
content**. That is the honest report — there was no answer yet — and it is why `split_think`
has to be told which mode it is in rather than guessing from the text.

Only `"high"` maps to `Reasoning Effort: High`; every other value maps to `Max`. That is the
template's own `capitalize` of its default, not a shortcut.

## Tool calling

Supported, and it is the checkpoint's own — hand-ported from `chat_template.jinja` with the
rest of the framing. Declarations go out as the template's `# Tools` system turn, calls come
back as `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>` and are
parsed into OpenAI `tool_calls`, and results return as
`<|observation|><tool_response>…</tool_response>` — one observation per *consecutive run* of
results, not one per result.

`finish_reason` becomes `tool_calls`, which outranks `stop`: an agent loop branches on that
field, and `stop` reads as "done talking to you". `length` still outranks both, because a
call cut off by the token budget may be incomplete and `tool_calls` would assert it is not.

**Prose streams; calls do not.** The content channel is cut at the first `<tool_call>`, and a
trailing *partial* marker is held back too — mid-generation the text can end `…<tool_ca`, and
emitting that would leak a fragment the next token turns into a marker, after which `content`
would have to shrink. A delta stream cannot express shrinking. The calls then arrive as one
structured delta with per-call `index`.

`tool_choice` accepts `"auto"` and `"none"`. `"none"` is honoured by dropping the
declarations, which genuinely prevents calls. `"required"` and a named function are **refused
with a 400**: nothing here can constrain decoding, and answering prose to a client that
demanded a call would look like compliance.

## Deliberately absent

| | why |
|---|---|
| **Sampling** | The engine is greedy argmax and every number in `../measurement/benchmarks.md` is measured that way. `temperature`/`top_p` are accepted and IGNORED with one warning per process — dropping them silently would leave a client believing its own determinism story. |
| **`/v1/completions`** | A raw, unframed prompt leaves the model outside an assistant turn, where its EOS ids are unreachable, so it runs to the token limit and then loops. That endpoint would serve the documented degeneration failure by construction. |
| **Paging** | `--ctx` allocates the KV slabs once at startup (~51 KB/token on top of the expert pool). A conversation that does not fit is a 400, never a silent truncation. |
| **Auth, TLS, batching, multi-model** | llama-swap owns swapping and fronting; the bind is loopback-only. |

An image part in a `content` array becomes the template's own "unable to process this image"
reminder rather than being dropped — a silent drop is what makes the model describe pictures
it never received.

The per-request log carries the same degeneration check `-bench` does. A looped response is a
broken one and it benchmarks *faster*, so server mode is not allowed to be the path that
hides it.

## The chat template, and why it has a test

`artifact::tokenizer::encode_chat_turns` is a **hand-port of the fp8 checkpoint's
`chat_template.jinja`** — there is no Jinja engine here. The converted artifact does **not**
carry the template (only `tokenizer.json` and `generation_config.json` survive conversion);
the source does, at `manifest.json`'s `i4_source.src`.

Nothing coupled the port to its source, so it drifted:

> **CORRECTED 2026-08-01.** `encode_chat` emitted GLM-4's framing — `<|role|>\n{content}`,
> ending at `<|assistant|>\n` — for this checkpoint, which has **no separator after the role
> token** and ends at `<|assistant|><think></think>`. Every `-bench` run before that date was
> one token off-template per turn and carried no thinking prefill. The error came from
> reasoning about GLM's turn structure from memory instead of reading the `.jinja` in the
> source checkpoint. See the STATE block in `../measurement/benchmarks.md` for what it
> invalidates: free-running *text*, acceptance and hit rates all move; `--ppl` does not,
> because it scores a corpus through `encode` and never the chat framing.

`tests/artifact.rs::chat_framing_matches_the_checkpoint_template` and
`::tool_framing_matches_the_checkpoint_template` are now that coupling. They render the built
ids back to a string and compare against the template's output **literally**, with the
expected strings written out in full rather than built from the code's own pieces — so the
test cannot drift with the thing it checks. Both skip without `RIVOLI_ARTIFACT`.

> `serde_json`'s `preserve_order` feature is **required, not a preference**. The template
> renders each tool schema by iterating the object as the client sent it; the default `Map`
> is a `BTreeMap`, so `{"name":…,"description":…}` came out alphabetized as
> `{"description":…,"name":…}` — a different token sequence from the one the model was
> trained on. The tool framing test caught it on its first run. The same test pins the
> `", "` / `": "` spacing, which is Python's `json.dumps` and therefore Jinja's `tojson`;
> serde_json's compact form tokenizes differently, which is what `PythonSpacing` exists for.
