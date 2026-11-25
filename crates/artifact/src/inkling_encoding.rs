//! Inkling 2's chat framing — a hand-port of the checkpoint's `chat_template.jinja`, which
//! ships beside the tokenizer at `docs/measurement/inkling2-reference/mini/chat_template.jinja`.
//!
//! **Why a hand-port at all.** Nothing in this engine speaks Jinja, and adding an engine for one
//! file is a dependency for what is ultimately string concatenation. Every other model here is
//! hand-ported the same way ([`crate::glimmer_encoding`], [`crate::qwen_encoding`],
//! [`crate::v4_encoding`]).
//!
//! **The output is a STRING, not ids**, exactly as for the other three: it is tokenized
//! afterwards by [`crate::tiktoken::Vocab`], whose inline-specials shape is what makes every
//! marker this framing emits a single id (the 60 inline specials sit at ids 199,998..=200,057 —
//! see that module's header). The id pin lives in `crates/artifact/tests/inkling_tokenizer.rs`,
//! which renders a one-turn prompt here and compares the ids against the first-party
//! measurement.
//!
//! # What this port does NOT reproduce, and why
//!
//! Recorded here rather than left to a reader, because "the reference agrees" is the
//! invented-measurement class this repo punishes. Each of these is a deliberate scope line for
//! Inkling 2's first decode engine, not a misreading:
//!
//! - **`tools`** — the template's `<|message_system|>tool_declare<|content_xml|>…` preamble — is
//!   not rendered. Nothing in this engine can read a tool call back yet (there is no
//!   `<|content_invoke_tool_json|>` parser), so advertising the syntax would turn every tool
//!   request into prose with `finish_reason: "stop"`; the same scope argument
//!   `serve::glimmer::glimmer_prompt` carries.
//! - **`tool_calls` blocks** are not emitted on an assistant turn, for the same reason. The key
//!   IS read, because the template's empty-content rule branches on it (`content` is skipped for
//!   an assistant turn that has calls), and dropping content without checking it would silently
//!   change other turns.
//! - **The `name` suffix on a `tool` or `assistant` header** is not appended. Roles frame as
//!   their bare `<|message_*|>` token; a tool result therefore loses its attribution rather than
//!   claiming a wrong one.
//! - **Image parts are REFUSED**, where the template renders `<|content_image|>` for a
//!   user/tool part. That is the owner's text-only scope decision for this engine: a port that
//!   emitted the placeholder would claim vision capability the decode path does not have.
//!
//! # Effort
//!
//! `reasoning_effort` may be a word from the template's own map, a NUMBER (the template's
//! non-string arm), or — a deliberate EXTENSION — a numeric STRING such as `"0.5"`, which the
//! Jinja refuses as `Unknown reasoning_effort`. Servers forward OpenAI's `reasoning_effort`,
//! which is a string, so refusing a numeric one would be a wire-format trap; the value is
//! validated in `[0.0, 0.99]` and rendered with Python's shortest-round-trip decimal form, which
//! is exactly what `{{ num }}` produces for the map's values (`0.1`, `0.7`, `0.99`, and `0` for
//! zero).

use anyhow::{Result, bail, ensure};
use serde_json::Value;

use crate::tokenizer::json_truthy;

/// `<|message_user|>`, id 200000.
pub const MESSAGE_USER: &str = "<|message_user|>";
/// `<|message_model|>`, id 200001.
pub const MESSAGE_MODEL: &str = "<|message_model|>";
/// `<|message_system|>`, id 200002.
pub const MESSAGE_SYSTEM: &str = "<|message_system|>";
/// `<|message_tool|>`, id 200003.
pub const MESSAGE_TOOL: &str = "<|message_tool|>";
/// `<|content_text|>`, id 200004.
pub const CONTENT_TEXT: &str = "<|content_text|>";
/// `<|content_thinking|>`, id 200008.
pub const CONTENT_THINKING: &str = "<|content_thinking|>";
/// `<|end_message|>`, id 200010.
pub const END_MESSAGE: &str = "<|end_message|>";
/// `<|content_model_end_sampling|>`, id 200006 — the checkpoint's `eos_token`.
pub const END_SAMPLING: &str = "<|content_model_end_sampling|>";

/// `reasoning_effort` when the caller names none — the template's `else 0.9`.
const DEFAULT_EFFORT: f64 = 0.9;

/// What the caller controls, mapping onto the template's one kwarg.
#[derive(Debug, Clone, Copy, Default)]
pub struct InklingChatOpts<'a> {
    /// `reasoning_effort`; `None` renders the template's default of `0.9`.
    pub reasoning_effort: Option<&'a str>,
}

/// Render an OpenAI-shaped `messages` array as Inkling 2's prompt string.
///
/// The effort block goes **once, before the first non-system message** — or at the very end when
/// every turn is a system turn, which is the template's `state.effort_emitted` rule and the
/// reason a loop that emitted it per message would be wrong on a system-first conversation.
/// One message's parsed role and its header token.
#[derive(Clone, Copy)]
struct Role {
    rtok: &'static str,
    is_system: bool,
    is_assistant: bool,
}

/// Render an OpenAI-shaped `messages` array as Inkling 2's prompt string.
///
/// The effort block goes **once, before the first non-system message** — or at the very end when
/// every turn is a system turn, which is the template's `state.effort_emitted` rule and the
/// reason a loop that emitted it per message would be wrong on a system-first conversation.
pub fn render(messages: &[Value], opts: &InklingChatOpts<'_>) -> Result<String> {
    let mut out = String::new();
    let mut emitted = false;
    for message in messages {
        let role = role_of(message)?;
        if !emitted && !role.is_system {
            out.push_str(&effort_block(opts)?);
            emitted = true;
        }
        push_message(&mut out, message, role)?;
    }
    if !emitted {
        out.push_str(&effort_block(opts)?);
    }
    Ok(out)
}

/// A message's role and its header token, refusing an unknown role.
///
/// The template raises for any role outside its four-entry map — there is no `else` arm — so a
/// `developer` turn is a refusal here and `serve::inkling::inkling_prompt` rewrites it to
/// `system` before rendering, exactly as the Glimmer door does.
fn role_of(message: &Value) -> Result<Role> {
    let name = message.get("role").and_then(Value::as_str).unwrap_or("");
    match name {
        "user" => Ok(Role {
            rtok: MESSAGE_USER,
            is_system: false,
            is_assistant: false,
        }),
        "assistant" => Ok(Role {
            rtok: MESSAGE_MODEL,
            is_system: false,
            is_assistant: true,
        }),
        "system" => Ok(Role {
            rtok: MESSAGE_SYSTEM,
            is_system: true,
            is_assistant: false,
        }),
        "tool" => Ok(Role {
            rtok: MESSAGE_TOOL,
            is_system: false,
            is_assistant: false,
        }),
        _ => bail!("Unknown message role: {name}"),
    }
}
/// The synthesised effort block: `<|message_system|><|content_text|>Thinking effort level: X…`.
fn effort_block(opts: &InklingChatOpts<'_>) -> Result<String> {
    let num = effort_number(opts.reasoning_effort)?;
    // `{{ num }}` for a Python float, with the template's zero special case. Non-zero values in
    // range are short decimals whose shortest round-trip form is identical in Rust and CPython.
    let shown = if num == 0.0 {
        "0".to_string()
    } else {
        format!("{num}")
    };
    Ok(format!(
        "{MESSAGE_SYSTEM}{CONTENT_TEXT}Thinking effort level: {shown}{END_MESSAGE}"
    ))
}

/// The effort as a number, from a word, a numeric string, or nothing.
fn effort_number(effort: Option<&str>) -> Result<f64> {
    let Some(raw) = effort else {
        return Ok(DEFAULT_EFFORT);
    };
    let key = raw.trim();
    if let Some(num) = named_effort(key) {
        return Ok(num);
    }
    if let Ok(num) = key.parse::<f64>() {
        ensure!(
            (0.0..=0.99).contains(&num),
            "reasoning_effort must be in [0.0, 0.99], got {raw:?}"
        );
        return Ok(num);
    }
    bail!("Unknown reasoning_effort: {raw}")
}

/// The template's `effort_map`, verbatim.
fn named_effort(key: &str) -> Option<f64> {
    Some(match key {
        "none" => 0.0,
        "minimal" => 0.1,
        "low" => 0.2,
        "medium" => 0.7,
        "high" => 0.9,
        "xhigh" | "max" => 0.99,
        _ => return None,
    })
}

/// One message's rendered blocks: reasoning, content, then the assistant tail.
fn push_message(out: &mut String, message: &Value, role: Role) -> Result<()> {
    if role.is_assistant
        && let Some(reasoning) = reasoning_text(message)?
    {
        push_block(out, (role.rtok, CONTENT_THINKING), reasoning);
    }
    push_content(out, message, role)?;
    if role.is_assistant {
        out.push_str(END_SAMPLING);
    }
    Ok(())
}

/// `reasoning_content` when it is a non-empty string; `Ok(None)` when absent or empty.
///
/// The template only looks at it on an assistant turn, and so does [`push_message`]. A non-string
/// value is refused rather than stringified: Jinja's `~` would call Python `str()` on it, and a
/// reasoning channel is not a place to invent a rendering for a shape no client sends.
fn reasoning_text(message: &Value) -> Result<Option<&str>> {
    match message.get("reasoning_content") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) if s.is_empty() => Ok(None),
        Some(Value::String(s)) => Ok(Some(s)),
        Some(_) => bail!("reasoning_content must be a string"),
    }
}

/// Whether an assistant message carries channels besides its text — the template's
/// `message.tool_calls or message.reasoning_content`.
fn carries_calls(message: &Value) -> bool {
    ["tool_calls", "reasoning_content"]
        .iter()
        .any(|k| message.get(*k).is_some_and(json_truthy))
}

/// `content`, as the template's two arms see it: a string, or a parts array.
///
/// The empty-string arm is the template's own conditional, not an optimisation: an assistant
/// turn that has a reasoning channel or tool calls emits NO text block when its `content` is
/// empty, while an empty user message still frames `<|content_text|><|end_message|>`.
fn push_content(out: &mut String, message: &Value, role: Role) -> Result<()> {
    match message.get("content") {
        None | Some(Value::Null) => Ok(()),
        Some(Value::String(s)) => {
            let skip = s.is_empty() && role.is_assistant;
            if !(skip && carries_calls(message)) {
                push_block(out, (role.rtok, CONTENT_TEXT), s);
            }
            Ok(())
        }
        Some(Value::Array(parts)) => push_parts(out, parts, role),
        Some(_) => bail!("message content must be a string or an array of parts"),
    }
}

/// One parts array: text parts, or a loud refusal for the media kinds this engine cannot take.
fn push_parts(out: &mut String, parts: &[Value], role: Role) -> Result<()> {
    for part in parts {
        if let Value::String(s) = part {
            push_block(out, (role.rtok, CONTENT_TEXT), s);
            continue;
        }
        match part.get("type").and_then(Value::as_str) {
            None | Some("text" | "input_text") => {
                let text = part.get("text").and_then(Value::as_str).unwrap_or("");
                push_block(out, (role.rtok, CONTENT_TEXT), text);
            }
            Some("image" | "input_image" | "image_url") => {
                bail!("Inkling 2 v1 is text-only: image content is not supported")
            }
            Some("audio" | "input_audio" | "audio_url") => {
                bail!("Inkling 2 does not accept audio input")
            }
            Some(other) => bail!("Unsupported content part type: {other}"),
        }
    }
    Ok(())
}

/// One `<marker>body<|end_message|>` block, under whatever header token the message frames with.
fn push_block(out: &mut String, (rtok, marker): (&str, &str), body: &str) {
    out.push_str(rtok);
    out.push_str(marker);
    out.push_str(body);
    out.push_str(END_MESSAGE);
}

#[cfg(test)]
mod tests {
    // A broken fixture IS the report: panicking here is the intended failure mode.
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use serde_json::{Value, json};

    use super::{
        CONTENT_TEXT, CONTENT_THINKING, END_MESSAGE, END_SAMPLING, InklingChatOpts, MESSAGE_MODEL,
        MESSAGE_SYSTEM, MESSAGE_USER, render,
    };

    enum Tag {
        Effort,
        Sys,
        User,
        Think,
        Answer,
    }

    fn seg(tag: Tag, body: &str) -> String {
        match tag {
            Tag::Effort => {
                format!("{MESSAGE_SYSTEM}{CONTENT_TEXT}Thinking effort level: {body}{END_MESSAGE}")
            }
            Tag::Sys => format!("{MESSAGE_SYSTEM}{CONTENT_TEXT}{body}{END_MESSAGE}"),
            Tag::User => format!("{MESSAGE_USER}{CONTENT_TEXT}{body}{END_MESSAGE}"),
            Tag::Think => format!("{MESSAGE_MODEL}{CONTENT_THINKING}{body}{END_MESSAGE}"),
            Tag::Answer => format!("{MESSAGE_MODEL}{CONTENT_TEXT}{body}{END_MESSAGE}"),
        }
    }

    fn turn(role: &str, content: &str) -> Value {
        json!({"role": role, "content": content})
    }

    fn attempt(messages: &[Value], opts: InklingChatOpts<'_>) -> Result<String, String> {
        render(messages, &opts).map_err(|err| err.to_string())
    }

    fn check(messages: &[Value], opts: InklingChatOpts<'_>, want: &[String]) {
        let got = attempt(messages, opts).expect("this fixture must render");
        assert_eq!(got, want.concat(), "{messages:?}");
    }

    fn refusal(messages: &[Value], opts: InklingChatOpts<'_>) -> String {
        attempt(messages, opts).expect_err("this input must be refused")
    }

    fn opts(reasoning_effort: Option<&str>) -> InklingChatOpts<'_> {
        InklingChatOpts { reasoning_effort }
    }

    /// The effort block for every level the template maps, its default, and the numeric-STRING
    /// extension: `0.5` renders as `0.5`, not `0.500000`, and `" 0 "` is trimmed.
    #[test]
    fn the_effort_block_precedes_every_first_non_system_turn() {
        for (word, shown) in [
            (Some("none"), "0"),
            (Some("medium"), "0.7"),
            (Some("max"), "0.99"),
            (Some("0.5"), "0.5"),
            (Some(" 0 "), "0"),
            (None, "0.9"),
        ] {
            let want = [seg(Tag::Effort, shown), seg(Tag::User, "Hi")];
            check(&[turn("user", "Hi")], opts(word), &want);
        }
    }

    /// The effort block is emitted ONCE: after a leading system turn and before the first user
    /// turn, or at the END when every turn is a system turn (the template's own
    /// `if not state.effort_emitted` tail).
    #[test]
    fn the_effort_block_lands_after_a_system_turn_or_at_the_end() {
        let leading = [
            seg(Tag::Sys, "be brief"),
            seg(Tag::Effort, "0.9"),
            seg(Tag::User, "Hi"),
        ];
        check(
            &[turn("system", "be brief"), turn("user", "Hi")],
            opts(Some("high")),
            &leading,
        );
        let only_system = [seg(Tag::Sys, "s"), seg(Tag::Effort, "0.9")];
        check(&[turn("system", "s")], opts(None), &only_system);
    }

    /// Reasoning renders before content, the turn closes with `<|content_model_end_sampling|>`,
    /// and an empty assistant `content` is skipped exactly when the turn has its own channel.
    #[test]
    fn assistant_channels_render_in_order_and_empty_text_is_conditional() {
        let both = [
            seg(Tag::Effort, "0.1"),
            seg(Tag::User, "2+2?"),
            seg(Tag::Think, "four"),
            seg(Tag::Answer, "4"),
            END_SAMPLING.to_string(),
        ];
        let r_turn = json!({"role": "assistant", "reasoning_content": "four", "content": "4"});
        check(
            &[turn("user", "2+2?"), r_turn],
            opts(Some("minimal")),
            &both,
        );
        let reasoned = [
            seg(Tag::Effort, "0.9"),
            seg(Tag::Think, "r"),
            END_SAMPLING.to_string(),
        ];
        let r_only = json!({"role": "assistant", "reasoning_content": "r", "content": ""});
        check(&[r_only], opts(None), &reasoned);
        let bare = [
            seg(Tag::Effort, "0.9"),
            seg(Tag::Answer, ""),
            END_SAMPLING.to_string(),
        ];
        check(&[turn("assistant", "")], opts(None), &bare);
    }

    /// Text parts and the `type`-absent default render; a string part is text too.
    #[test]
    fn text_parts_render_and_an_absent_type_is_text() {
        let content = json!([
            "raw",
            {"type": "text", "text": "one"},
            {"text": "two"},
            {"type": "input_text", "text": "three"}
        ]);
        let want = [
            seg(Tag::Effort, "0"),
            seg(Tag::User, "raw"),
            seg(Tag::User, "one"),
            seg(Tag::User, "two"),
            seg(Tag::User, "three"),
        ];
        check(
            &[json!({"role": "user", "content": content})],
            opts(Some("none")),
            &want,
        );
    }

    /// Every out-of-scope part kind, with the exact refusal it earns.
    #[test]
    fn unsupported_parts_are_refused_with_their_messages() {
        for (kind, want) in [
            (
                "image",
                "Inkling 2 v1 is text-only: image content is not supported",
            ),
            (
                "input_image",
                "Inkling 2 v1 is text-only: image content is not supported",
            ),
            (
                "image_url",
                "Inkling 2 v1 is text-only: image content is not supported",
            ),
            ("input_audio", "Inkling 2 does not accept audio input"),
            ("video", "Unsupported content part type: video"),
        ] {
            let messages = [json!({"role": "user", "content": [{"type": kind}]})];
            assert_eq!(refusal(&messages, opts(None)), want, "part {kind:?}");
        }
    }

    /// The template has no arm for `developer`, so the role is refused rather than dropped.
    #[test]
    fn an_unknown_role_is_refused_rather_than_dropped() {
        let messages = [turn("developer", "x")];
        assert_eq!(
            refusal(&messages, opts(None)),
            "Unknown message role: developer"
        );
    }

    /// An effort outside the map, out of range, or not a number is refused by name.
    #[test]
    fn out_of_range_and_unknown_efforts_are_refused() {
        for word in ["1.5", "-0.1", "NaN", "extreme"] {
            let got = refusal(&[turn("user", "Hi")], opts(Some(word)));
            assert!(
                got.starts_with("reasoning_effort must be in [0.0, 0.99]")
                    || got.starts_with("Unknown reasoning_effort"),
                "unexpected message for {word:?}: {got}"
            );
        }
    }
}
