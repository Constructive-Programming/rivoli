//! The input side: the typed conversation, and the ONE boundary where OpenAI-shaped JSON
//! becomes it.
//!
//! Split out of the reference's single 2822-line module (`old:src/artifact/dsv4_encoding.rs`)
//! by cohesion, under the 800-line cap. The seam is the one the reference already draws with a
//! banner comment: everything here answers "what did the caller send", and nothing here knows
//! what a prompt looks like. [`super::render`] is the other half and imports these types; the
//! dependency runs one way only.
//!
//! **Every refusal below carries the measurement that produced it.** The header of
//! [`super`] holds the accept/reject table these implement, and
//! `super::tests::boundary::the_over_refusal_table_is_real` executes every row of it.

use super::Task;
use crate::tokenizer::json_truthy;
use anyhow::{Result, bail, ensure};
use serde_json::Value;

// ============================================================
// Input messages
// ============================================================

/// One tool the model may call: the OpenAI wrapper's **`function` object, verbatim**.
///
/// Verbatim matters. `render_tools` serializes this straight into the system prompt, so the
/// key order the client sent is the key order the model sees — which is why `serde_json`
/// carries `preserve_order` in `Cargo.toml`. Unlike GLM's path there is no `defer_loading`
/// or `strict` stripping; the reference passes the object through untouched.
#[derive(Debug, Clone)]
pub struct Tool(pub(super) Value);

impl Tool {
    /// Parse one entry of an OpenAI `tools` array.
    ///
    /// The reference indexes `tool["function"]` unconditionally, so a bare function object
    /// (which GLM's path tolerates) is a `KeyError` there and an `Err` here rather than a
    /// guess.
    pub fn from_openai(tool: &Value) -> Result<Self> {
        let f = tool
            .get("function")
            .filter(|f| f.is_object())
            .ok_or_else(|| anyhow::anyhow!("tool entry has no `function` object: {tool}"))?;
        Ok(Self(f.clone()))
    }

    /// A function object that did NOT arrive inside an OpenAI `{"type","function"}` wrapper.
    ///
    /// Needed because [`Instructions::tools`] is a public field: without this, a caller
    /// building `Instructions` by hand — rather than through [`messages_from_openai`] —
    /// could not fill it without synthesising a wrapper it does not have. Stored verbatim,
    /// exactly as `from_openai` stores the unwrapped object.
    #[must_use]
    pub const fn from_function(function: Value) -> Self {
        Self(function)
    }
}

/// One call the assistant made, in the OpenAI shape: `arguments` is a JSON **string**.
#[derive(Debug, Clone)]
pub struct ToolCall {
    /// Matched against a later tool result's `tool_call_id` to order results by call order.
    /// Empty is allowed — the reference's `tc.get("id") or …get("id", "")` tolerates it, and
    /// then the result sorts as if its index were 0.
    pub id: String,
    pub name: String,
    /// OpenAI says a JSON **string**; this is a `Value` because the reference copies the
    /// field verbatim and only tries `json.loads` at render time, inside a `try` with a bare
    /// `except Exception`. A client that sends an *object* here — a common deviation — makes
    /// `json.loads` raise `TypeError`, which that `except` swallows, and the whole object
    /// then renders as one `arguments` parameter with `string="false"`. Refusing it, or
    /// re-encoding it to a string so it parses per-key, would both be wrong.
    /// See `super::render::push_dsml_parameters`.
    pub arguments: Value,
}

/// A `system` or `developer` turn: instructions, plus the two things only they may declare.
#[derive(Debug, Clone, Default)]
pub struct Instructions {
    pub content: String,
    /// Declaring ANY tool anywhere in the conversation turns `drop_thinking` off — a
    /// tool-calling conversation needs every turn's reasoning to track multi-step work.
    pub tools: Vec<Tool>,
    /// Rendered as a `## Response Format:` block. Skipped when falsy in Python's sense
    /// (`None`, `{}`, `[]`, `""`, `0`), which is what `if response_format:` means.
    pub response_format: Option<Value>,
}

/// An assistant turn: what it reasoned, what it said, and what it called.
///
/// One struct shared by [`Body::Assistant`] and the post-merge form, because the merge does
/// not touch an assistant turn — a second copy of these four fields would be a `build.rs`
/// duplication error and, worse, two places to add the fifth.
#[derive(Debug, Clone)]
pub struct AssistantTurn {
    pub content: String,
    /// The `<think>…</think>` block. Dropped from turns before the last user message unless
    /// tools are in play — see [`super::EncodeOpts::drop_thinking`].
    pub reasoning_content: String,
    pub tool_calls: Vec<ToolCall>,
    /// Leave this turn's EOS off — for continuing a partial assistant message.
    pub wo_eos: bool,
}

/// What kind of turn this is. The per-role fields live in the variant, so "tools on a user
/// message" and "reasoning on a system message" are unrepresentable rather than ignored.
#[derive(Debug, Clone)]
pub enum Body {
    System(Instructions),
    /// Framed as a USER turn that happens to carry tools. Internal to DeepSeek's search
    /// agent — their API rejects it — but the reference encodes it and test case 3 uses it,
    /// so it is here.
    Developer(Instructions),
    User {
        content: String,
    },
    /// Merged into the preceding user turn as a `<tool_result>` block; this role never
    /// reaches the renderer — `merge_tool_messages` returns a type that has no variant for
    /// it, which is how the reference's runtime `NotImplementedError` becomes a compile
    /// error.
    ///
    /// `content` is already the rendered text — see [`tool_content`], which flattens the block
    /// list a client may send.
    Tool {
        tool_call_id: String,
        content: String,
    },
    LatestReminder {
        content: String,
    },
    Assistant(AssistantTurn),
}

/// One message plus its optional quick-instruction task.
///
/// `task` sits beside the body rather than inside a variant because the reference reads it
/// off *any* message: `action`/`query`/`authority`/`domain`/`read_url` on a user turn,
/// `title` on an assistant turn.
#[derive(Debug, Clone)]
pub struct Message {
    pub body: Body,
    pub task: Option<Task>,
}

impl Message {
    /// A message with no task — every ordinary turn.
    #[must_use]
    pub const fn new(body: Body) -> Self {
        Self { body, task: None }
    }

    #[must_use]
    pub fn user(content: impl Into<String>) -> Self {
        Self::new(Body::User {
            content: content.into(),
        })
    }
}

// ============================================================
// The boundary: OpenAI-shaped JSON in, typed messages out
// ============================================================

impl Task {
    /// The `"task"` field's six legal values.
    ///
    /// Refused at the boundary, which is STRICTER than the reference on one path: its
    /// `assert task in VALID_TASKS` sits after `render_message`'s early return, so a task on
    /// a message followed by an ordinary user turn is never validated and silently does
    /// nothing there. A misspelled task that quietly stops being a task is the failure this
    /// module exists to make loud.
    fn parse(name: &str) -> Result<Self> {
        match name {
            "action" => Ok(Self::Action),
            "query" => Ok(Self::Query),
            "authority" => Ok(Self::Authority),
            "domain" => Ok(Self::Domain),
            "title" => Ok(Self::Title),
            "read_url" => Ok(Self::ReadUrl),
            other => bail!(
                "invalid task: '{other}'. Valid tasks are: action, query, authority, domain, \
                 title, read_url"
            ),
        }
    }
}

impl ToolCall {
    fn from_openai(tc: &Value) -> Result<Self> {
        let f = tc
            .get("function")
            .ok_or_else(|| anyhow::anyhow!("tool call has no `function`: {tc}"))?;
        Ok(Self {
            // `id` on the wrapper, else on the function object — the two places the
            // reference's `sort_tool_results_by_call_order` looks, in that order. Absent is
            // allowed: it only costs the result its sort key. A present-but-non-string id
            // is refused rather than read through `as_str()`, because Python keys its order
            // dict on the RAW value — an id of `123` matched by a `tool_call_id` of `123` is
            // a live sort key there, and dropping both ends to `""` would silently reorder
            // the `<tool_result>` blocks. Both ends are refused, so that pair cannot arise.
            //
            // The fallback SHORT-CIRCUITS, like the `or` it ports: when the wrapper carries a
            // usable id the function object's is never read, so junk in `function.id` beside
            // a good wrapper id must not be refused.
            id: match optional_str(tc, "id")? {
                Some(s) if !s.is_empty() => s.to_string(),
                _ => optional_str(f, "id")?.unwrap_or_default().to_string(),
            },
            name: optional_str(f, "name")?
                .ok_or_else(|| anyhow::anyhow!("tool call `function.name` must be a string: {tc}"))?
                .to_string(),
            // Verbatim, whatever type it is — see the field's doc. Required: the reference
            // indexes `tool_call["function"]["arguments"]` and KeyErrors if it is absent.
            arguments: f
                .get("arguments")
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("tool call has no `function.arguments`: {tc}"))?,
        })
    }
}

/// One entry of a **`tool` message's** content list — the reference's INNER walker, the
/// `isinstance(tool_content, list)` loop nested inside its `tool_result` arm.
///
/// Four arms, each a measurement rather than a reading of the README: the reference walks
/// these by hand instead of formatting them, so every shape has its own observable
/// behaviour. Measured 2026-08-06. The OUTER walker is [`content_block_text`], which has one
/// arm more; they are separate functions there and separate here.
fn tool_block_text(b: &Value) -> Result<String> {
    // A block that is not an OBJECT is `AttributeError: 'str' object has no attribute 'get'`
    // there — it is not an unsupported block, it is not a block.
    let b = b
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("a content block must be an object, got {b}"))?;
    match b.get("type") {
        Some(Value::String(ty)) if ty == "text" => Ok(match b.get("text") {
            // `.get("text", "")` — absent is `""` on both sides.
            None => String::new(),
            Some(Value::String(s)) => s.clone(),
            // …but a NULL or non-string `text` reaches `"\n\n".join` and raises TypeError
            // there, where `unwrap_or_default()` quietly emptied the block here. This is the
            // shape a client sends when a tool returned nothing. Review finding, 2026-08-06.
            Some(other) => bail!(
                "a text block's `text` must be a string, got {other}; the reference raises \
                 TypeError on this"
            ),
        }),
        // A non-text block is NAMED, not dropped — the reference's own placeholder, because
        // losing the fact that something was there is worse than saying so.
        Some(Value::String(ty)) => Ok(format!("[Unsupported {ty}]")),
        // Absent and null both spell the word `None`: Python's f-string interpolating a
        // missing key. Reproduced, per this module's refuse-vs-reproduce rule.
        None | Some(Value::Null) => Ok("[Unsupported None]".to_string()),
        // Any other `type` interpolates Python's `str()`/`repr()`: `5` gives
        // `[Unsupported 5]`, but `{"a": 1}` gives `[Unsupported {'a': 1}]`, single quotes and
        // all. Refused rather than guessed at. A binding rather than four spelled-out arms:
        // `Value` is `serde_json`'s, not ours.
        Some(ty) => bail!(
            "a content block's `type` must be a string, got {ty}; the reference interpolates \
             Python's repr of it"
        ),
    }
}

/// One entry of a `user`/`system`/`assistant` **array-form `content`** — the reference's
/// OUTER walker, `render_message`'s `content_blocks` branch.
///
/// Differs from the inner walker by exactly one arm, so it delegates for the rest: a
/// `tool_result` block keeps its PAYLOAD, `<tool_result>…</tool_result>`, where the inner
/// walker would only name the type and drop the content. Getting that wrong was the shape of
/// a review finding on 2026-08-06 — the comment cited this branch as the rule and the code
/// ran the other one.
fn content_block_text(b: &Value) -> Result<String> {
    let is_tool_result = b.get("type").and_then(Value::as_str) == Some("tool_result");
    if !is_tool_result {
        return tool_block_text(b);
    }
    Ok(format!(
        "<tool_result>{}</tool_result>",
        tool_content_text(b.get("content"))?
    ))
}

/// `\n\n`-join a block list through `walk`. One function because both walkers are folded
/// identically and the two four-line copies were a jscpd tripwire waiting on one more line.
fn join_blocks(blocks: &[Value], walk: fn(&Value) -> Result<String>) -> Result<String> {
    Ok(blocks
        .iter()
        .map(walk)
        .collect::<Result<Vec<_>>>()?
        .join("\n\n"))
}

/// A `tool` message's `content`, rendered. A plain string passes through; the block list
/// some clients send is `\n\n`-joined here, at the boundary, because nothing downstream ever
/// looks at the structure.
fn tool_content(msg: &Value) -> Result<String> {
    tool_content_text(msg.get("content"))
}

/// The same, from the raw field — which is where a `tool_result` block inside an array-form
/// `content` carries it, rather than on a message. Takes `Option` because absent and null
/// differ here and nowhere else.
fn tool_content_text(content: Option<&Value>) -> Result<String> {
    Ok(match content {
        // Absent and NULL differ here, measured 2026-08-06: `msg.get("content", "")` yields
        // `""` when the key is missing and `None` when it is present-and-null, and only the
        // second reaches the template — as the word `None`. Same `str()` that makes a bare
        // scalar unguessable below, so it gets the same refusal rather than a quiet `""`.
        None => String::new(),
        // REPRODUCED, not refused. `None` is one determined four-character literal that
        // `tool_block_text` just above already emits for a block's absent `type`, and this is
        // the shape an OpenAI client sends when a tool returns nothing — so refusing it would
        // 400 on ordinary traffic the reference encodes fine. See the header's rule.
        // (Named, not counted: a line distance is the thing that rots on the next edit, and
        // this comment had already inverted once when the walker moved above.)
        Some(Value::Null) => "None".to_string(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(blocks)) => join_blocks(blocks, tool_block_text)?,
        // The reference formats a scalar straight into its string template, so `5` becomes
        // `<tool_result>5</tool_result>` — but `true` becomes `True` and `null` becomes
        // `None`, because that is Python's `str()` and not JSON. Refused rather than either
        // guessing which of those to emit or silently dropping the content, which is what an
        // `as_str().unwrap_or_default()` fallback did until review caught it 2026-08-05.
        Some(other) => bail!(
            "a `tool` message's `content` must be a string or a block list, got {other}; the \
             reference renders Python's `str()` of it, which is not JSON"
        ),
    })
}

/// The `content` of a `system`, `developer`, `user` or `assistant` turn: a string, or
/// OpenAI's array of content parts. Absent or null becomes `""` — the reference's
/// `content or ""`, which `developer` then rejects via its own non-empty assert.
///
/// Anything else is an ERROR rather than a silent `""` — `as_str().unwrap_or_default()` would
/// drop the user's entire question and leave `<｜User｜><｜Assistant｜><think>`, a prompt that
/// looks perfectly well formed. Found in review 2026-08-05.
fn text_content(msg: &Value) -> Result<String> {
    match msg.get("content") {
        None | Some(Value::Null) => Ok(String::new()),
        Some(Value::String(s)) => Ok(s.clone()),
        // OpenAI's ARRAY-OF-PARTS content, which is standard in their schema and which real
        // clients send constantly. Accepted, and walked by `content_block_text`.
        //
        // A DELIBERATE divergence, and the only one here that makes this port accept
        // something the reference does not encode identically. The reference has no coherent
        // handling of this shape: on a `user` turn it raises TypeError (so there are no
        // reference bytes to be faithful TO — this fills a hole rather than contradicting
        // one), and on `system`/`developer`/`assistant` it formats Python's list repr,
        // `[{'type': 'text', 'text': 'hi'}]`, single quotes and all, which cannot be a
        // sequence the model was trained on.
        //
        // The rule followed is `render_message`'s `content_blocks` branch. That branch is
        // NOT dead — `merge_tool_messages` attaches `content_blocks` to every user message,
        // so it is the live rendering path for every user turn in every conversation, which
        // is what `message_from_openai`'s `user` arm below already says. What is unreachable
        // is a CLIENT-supplied `content_blocks` key, because the merge rebuilds the dict and
        // discards it. So this is the reference's own live rule for a list of parts, reached
        // by a spelling it happens to reject. Measured 2026-08-06.
        Some(Value::Array(blocks)) => Ok(join_blocks(blocks, content_block_text)?),
        Some(other) => bail!(
            "`content` must be a string or an array of content parts, got {other}. The \
             reference's templates are string templates and would format Python's `str()` of \
             this into the prompt."
        ),
    }
}

/// One `Value` field that must be a string if it is present at all.
///
/// The difference from `as_str()` is the whole point: that silently reads a non-string as
/// absent, and for an id or a task "absent" is a *different prompt*, not an error.
fn optional_str<'v>(msg: &'v Value, key: &str) -> Result<Option<&'v str>> {
    match msg.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s)),
        Some(other) => bail!("`{key}` must be a string, got {other}"),
    }
}

/// One `Value` field that must be an array if it is present at all.
///
/// Absent is fine — the reference's `msg.get(...)` yields `None`, which is falsy. A TRUTHY
/// non-array raises there, when it is indexed; a FALSY one (`{}`, `""`, `0`) is skipped by
/// the `if tools:` guard and simply ignored. Both are refused here, so this is stricter than
/// the reference on the falsy case; the header's table carries the row.
fn array_field<'v>(msg: &'v Value, key: &str) -> Result<&'v [Value]> {
    match msg.get(key) {
        None | Some(Value::Null) => Ok(&[]),
        Some(Value::Array(a)) => Ok(a),
        Some(other) => bail!("`{key}` must be an array, got {other}"),
    }
}

/// The `tools` / `response_format` a system or developer turn may declare.
fn instructions_from_openai(msg: &Value) -> Result<Instructions> {
    Ok(Instructions {
        content: text_content(msg)?,
        tools: array_field(msg, "tools")?
            .iter()
            .map(Tool::from_openai)
            .collect::<Result<_>>()?,
        response_format: msg.get("response_format").cloned(),
    })
}

/// The body of one message, by role. Split out of [`message_from_openai`] so that the
/// role dispatch and the two whole-message checks that follow it (the stray-`tools` scan and
/// the task) are separate readings — the reference has them in one body, which is the shape
/// the code-health gate refuses.
fn body_from_openai(role: &str, msg: &Value) -> Result<Body> {
    Ok(match role {
        "system" => Body::System(instructions_from_openai(msg)?),
        "developer" => Body::Developer(instructions_from_openai(msg)?),
        "user" => Body::User {
            // A NULL `content` on a user turn is a TypeError in the reference, not an empty
            // turn: post-merge the live path is the `"\n\n".join` over `content_blocks`,
            // whose `text` is `None` — `render_message`'s `content or ""` is dead for this
            // role. Distinct from `assistant`, where OpenAI sends `content: null` routinely
            // alongside `tool_calls` and the reference does yield `""`, so that one must NOT
            // be refused. Both measured 2026-08-06. Review finding.
            content: {
                ensure!(
                    !matches!(msg.get("content"), Some(Value::Null)),
                    "`content` is null on a `user` message; the reference raises TypeError \
                     here (though it accepts null on an `assistant` turn, which is the \
                     ordinary shape beside `tool_calls`). Send \"\" for an empty turn."
                );
                text_content(msg)?
            },
        },
        // The only role with no `content or ""` anywhere on its path: absent AND null both
        // reach `"{content}".format(...)` as Python's `None` and print that word into the
        // prompt (measured 2026-08-06). Neither is a sequence the model was trained on, and a
        // silent `""` here would be a third behaviour that is nobody's.
        "latest_reminder" => Body::LatestReminder {
            content: match msg.get("content") {
                Some(Value::String(s)) => s.clone(),
                // Absent and null BOTH land here, unlike every other role — measured.
                None | Some(Value::Null) => "None".to_string(),
                Some(other) => bail!(
                    "a `latest_reminder`'s `content` must be a string, got {other}; the \
                     reference interpolates Python's repr of it"
                ),
            },
        },
        // The one role whose `content` is ALLOWED to be a block list — the reference walks it
        // (`if isinstance(tool_content, list)`) rather than formatting it into a template.
        "tool" => Body::Tool {
            // Same rule as a tool call's `id`, and for the same reason: it is the key the
            // result is matched on.
            tool_call_id: optional_str(msg, "tool_call_id")?
                .unwrap_or_default()
                .to_string(),
            content: tool_content(msg)?,
        },
        "assistant" => Body::Assistant(AssistantTurn {
            content: text_content(msg)?,
            // The last field that was read through `as_str()`, and the reference FORMATS a
            // truthy non-string one straight into the reasoning block: `reasoning_content:
            // 123` renders `<think>123</think>` there and rendered `<think></think>` here —
            // the whole block vanishing, silently. Review finding, 2026-08-06.
            //
            // A FALSY non-string (`0`, `false`, `[]`, `{}`) is `""` in the reference, via
            // `rc = reasoning_content or ""`, and is refused here anyway. That is deliberate
            // and it is the same call `text_content` makes: one rule for every content-ish
            // field — a non-string is refused — beats five role-conditional ones, because
            // the `or ""` guard is present for `system`/`assistant`, absent for
            // `latest_reminder`, and a raise for `user`. Both rows are in the header table.
            reasoning_content: optional_str(msg, "reasoning_content")?
                .unwrap_or_default()
                .to_string(),
            tool_calls: array_field(msg, "tool_calls")?
                .iter()
                .map(ToolCall::from_openai)
                .collect::<Result<_>>()?,
            // Truthiness, not `as_bool`: the reference tests `if wo_eos:`, so `1` counts.
            wo_eos: msg.get("wo_eos").is_some_and(json_truthy),
        }),
        other => bail!("unknown role: {other}"),
    })
}

/// A stray `tools` key on a role that renders nothing for it.
///
/// The reference's `any(m.get("tools") for m in full_messages)` runs over the MERGED
/// list, so which roles a stray `tools` key can affect is decided by what survives
/// `merge_tool_messages`. MEASURED against the reference 2026-08-06:
///
/// ```text
///   user, tool          rebuilt into a fresh dict, key dropped -> provably INERT
///                       (encoded output byte-identical with and without it)
///   assistant,          passed through untouched -> disables drop_thinking for the WHOLE
///   latest_reminder     conversation while rendering nothing, changing the thinking token
///                       on every earlier turn
/// ```
///
/// Only the second pair is refused. Refusing the first would reject input the reference
/// accepts and ignores — an over-strict check that a round-1 version of this had, caught
/// in review because it makes this port disagree in the *reject* direction.
fn ensure_tools_can_be_rendered(role: &str, body: &Body, msg: &Value) -> Result<()> {
    ensure!(
        !(matches!(body, Body::Assistant(_) | Body::LatestReminder { .. })
            && msg.get("tools").is_some_and(json_truthy)),
        "`tools` on a `{role}` message: only `system` and `developer` render a tools \
         block. The reference renders nothing for this one but still lets it disable \
         drop_thinking for every turn, which is a silent whole-conversation change."
    );
    Ok(())
}

/// Parse one OpenAI-shaped message object.
///
/// **This is the only place a raw `Value` becomes a [`Message`]**, so everything downstream
/// can stop asking whether a field is the right type. An unknown role is refused, matching
/// the reference's `NotImplementedError` — silently framing it as `user` (which GLM's path
/// does, with a warning) would put the wrong turn token in front of it.
pub fn message_from_openai(msg: &Value) -> Result<Message> {
    let role = msg
        .get("role")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("message has no `role`: {msg}"))?;
    let body = body_from_openai(role, msg)?;
    ensure_tools_can_be_rendered(role, &body, msg)?;
    // `as_str().and_then(...)` here would turn `{"task": 123}` into "no task" and encode a
    // perfectly ordinary prompt; the reference's `if task is not None` accepts the 123 and
    // then trips its `assert task in VALID_TASKS`. Review finding, 2026-08-06.
    let task = match msg.get("task") {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) => Some(Task::parse(s)?),
        Some(other) => bail!("`task` must be a string, got {other}"),
    };
    Ok(Message { body, task })
}

/// Parse an OpenAI `messages` array. One malformed turn fails the whole array — a prompt
/// assembled from the turns that happened to parse is a different conversation, not a
/// degraded one.
pub fn messages_from_openai(messages: &[Value]) -> Result<Vec<Message>> {
    messages
        .iter()
        .enumerate()
        // Which turn, by index. Without it a 40-turn conversation reports only that some
        // `content` was not a string.
        .map(|(i, m)| {
            anyhow::Context::with_context(message_from_openai(m), || format!("message {i}"))
        })
        .collect()
}
