//! Inkling 2's half of the OpenAI door: the request's prompt string and the reply's channels.
//!
//! Split out of [`super`] on the same argument as [`super::glimmer`]: the dispatch lives in
//! `serve/mod.rs`, and a model's framing does not belong in it. Pure functions of a request body
//! and a generated string — no tokenizer, no engine, no device — which is what lets the decisions
//! rivoli INVENTS here (the `developer` rewrite and the `think`/effort precedence) be gated
//! without a GPU.
//!
//! **Channel shape.** The template renders every assistant turn as a sequence of blocks:
//! `<|message_model|><|content_thinking|>…<|end_message|>` for reasoning,
//! `<|message_model|><|content_text|>…<|end_message|>` for the answer, and an optional trailing
//! `<|content_model_end_sampling|>`. The model emits the same bytes, so the channel is IN THE
//! TEXT and [`split_inkling`] needs no `thinking` flag — a flag would be a second authority on a
//! fact the bytes already carry.

use anyhow::{Context, Result};
use rivoli_artifact::inkling_encoding::{
    CONTENT_TEXT, CONTENT_THINKING, END_MESSAGE, END_SAMPLING, InklingChatOpts, MESSAGE_MODEL,
    MESSAGE_SYSTEM, MESSAGE_TOOL, MESSAGE_USER, render,
};
use serde_json::Value;

/// Every role tag a turn can open with, the model's last. A reply is read as unstructured text
/// only when no channel marker appeared, and any of these may still lead it — a model echoing
/// `<|message_system|>` must not leak the raw role marker into the client's screen.
const ROLE_MARKERS: [&str; 4] = [MESSAGE_SYSTEM, MESSAGE_USER, MESSAGE_TOOL, MESSAGE_MODEL];

/// Inkling 2's prompt STRING for one request — the two decisions serve makes that
/// `inkling_encoding::render` cannot make for itself.
///
/// **1. `developer` becomes `system`.** The template raises on any role outside its four-entry
/// map, so an unmapped OpenAI `developer` turn would be a 400 on a body every other endpoint
/// accepts; the rewrite makes it behave exactly like `system`, which is what the role means.
///
/// **2. `think == false` OUTRANKS an explicit effort**, the same decision — and for the same
/// reason — as [`super::glimmer::glimmer_prompt`]: a request carrying
/// `{"enable_thinking": false, "reasoning_effort": "high"}` arrives as `(false, Some("high"))`,
/// and the instruction is "do not think"; a strength word is what to do IF thinking.
///
/// **`tools` is deliberately not forwarded**: the template's `tool_declare` preamble advertises a
/// call syntax nothing here can read back, which would turn tool use into prose with
/// `finish_reason: "stop"`.
pub fn inkling_prompt(body: &Value, (think, effort): (bool, Option<&str>)) -> Result<String> {
    let messages: Vec<Value> = body
        .get("messages")
        .and_then(Value::as_array)
        .with_context(|| "an Inkling request needs a `messages` array")?
        .iter()
        .map(promote_developer)
        .collect();
    render(
        &messages,
        &InklingChatOpts {
            reasoning_effort: if think { effort } else { Some("none") },
        },
    )
}

/// One message with OpenAI's `developer` role rewritten to `system`.
///
/// A `map` over the array rather than an in-place loop, so the clone-and-rewrite is one named
/// step and the framing call below reads as a call rather than a mutation.
fn promote_developer(message: &Value) -> Value {
    let mut promoted = message.clone();
    if promoted.get("role").and_then(Value::as_str) == Some("developer") {
        promoted["role"] = Value::String("system".to_owned());
    }
    promoted
}

/// Which output channel a block belongs to.
#[derive(Clone, Copy)]
enum Channel {
    Thinking,
    Text,
}

/// Split an Inkling 2 generation into `(reasoning, content)`.
///
/// **Segments, walked in order.** Every block is a channel marker, a body, and
/// `<|end_message|>`; the walk starts at the first `<|content_thinking|>` or `<|content_text|>`
/// and discards whatever precedes it, because that is the `<|message_model|>` header (plus a
/// name, which this engine does not use). Bare channel markers — a model that omitted the
/// role tag — work identically.
///
/// **`complete` says whether `full` is the WHOLE generation, and it is load-bearing for
/// streaming.** A prefix can end mid-marker (`"…<|end_me"`) or mid-marker-name
/// (`"<|content_te"`), and emitting those bytes would put text in the stream the finished reply
/// does not contain — the same leak [`super::glimmer::split_glimmer`]'s marker hold exists to
/// stop. With `complete`, every byte inside a channel is the model's.
///
/// **No channel marker at all is the unstructured case.** A complete generation is the whole
/// reply and comes back as content; a PREFIX is held only while it could still BE a marker, so
/// an ordinary reply streams as it arrives.
/// Internal scanner holding the generation slice and completion flag so the channel helpers
/// share one cursor instead of repeating `(&str, bool)` arguments.
#[derive(Clone, Copy)]
struct Scan<'a> {
    text: &'a str,
    complete: bool,
}

impl<'a> Scan<'a> {
    fn next_channel(self) -> Option<(usize, Channel, usize)> {
        [
            (CONTENT_THINKING, Channel::Thinking),
            (CONTENT_TEXT, Channel::Text),
        ]
        .iter()
        .filter_map(|(marker, channel)| {
            self.text
                .find(marker)
                .map(|at| (at, *channel, marker.len()))
        })
        .min_by_key(|(at, _, _)| *at)
    }

    fn trim_partial(self) -> &'a str {
        if self.complete {
            return self.text;
        }
        let markers =
            ROLE_MARKERS
                .iter()
                .copied()
                .chain([END_MESSAGE, CONTENT_TEXT, CONTENT_THINKING]);
        self.text
            .rfind('<')
            .filter(|at| markers.clone().any(|m| m.starts_with(&self.text[*at..])))
            .map_or(self.text, |at| &self.text[..at])
    }

    fn channel_body(self) -> (&'a str, usize) {
        let end = [END_MESSAGE, CONTENT_THINKING, CONTENT_TEXT]
            .iter()
            .filter_map(|m| self.text.find(m))
            .min()
            .unwrap_or(self.text.len());
        let closed = self.text[end..].starts_with(END_MESSAGE);
        let piece = Scan {
            text: &self.text[..end],
            complete: self.complete,
        }
        .trim_partial();
        (piece, if closed { end + END_MESSAGE.len() } else { end })
    }

    fn unstructured(self) -> (String, String) {
        // Any role tag, not just the model's: a model echoing `<|message_system|>` with no
        // channel tag would otherwise put the raw marker on the user's screen.
        let text = ROLE_MARKERS
            .iter()
            .find_map(|m| self.text.strip_prefix(*m))
            .unwrap_or(self.text);
        if text.is_empty() {
            return (String::new(), String::new());
        }
        let maybe_marker = ROLE_MARKERS
            .iter()
            .copied()
            .chain([CONTENT_TEXT, CONTENT_THINKING])
            .any(|m| m.starts_with(text));
        if !self.complete && maybe_marker {
            return (String::new(), String::new());
        }
        let content = Scan {
            text,
            complete: self.complete,
        }
        .trim_partial();
        (String::new(), content.to_string())
    }
}

pub fn split_inkling(text: &str, complete: bool) -> (String, String) {
    let body = text.strip_suffix(END_SAMPLING).unwrap_or(text);
    let mut scan = Scan {
        text: body,
        complete,
    };
    let (mut reasoning, mut content) = (String::new(), String::new());
    let mut saw_channel = false;
    while let Some((at, channel, marker_len)) = scan.next_channel() {
        saw_channel = true;
        let tail = Scan {
            text: &scan.text[at + marker_len..],
            complete,
        };
        let (piece, consumed) = tail.channel_body();
        match channel {
            Channel::Thinking => reasoning.push_str(piece),
            Channel::Text => content.push_str(piece),
        }
        scan.text = &tail.text[consumed..];
    }
    if saw_channel {
        (reasoning, content)
    } else {
        Scan {
            text: body,
            complete,
        }
        .unstructured()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)] // a broken fixture IS the report

    use serde_json::json;

    use super::{inkling_prompt, split_inkling};

    /// A request message with its role and string content — the only fields these fixtures set.
    fn message(role: &str, content: &str) -> serde_json::Value {
        json!({"role": role, "content": content})
    }

    /// The full turn shape, complete: reasoning then answer then the end-of-sampling stop.
    #[test]
    fn a_thinking_then_text_generation_splits_into_its_two_channels() {
        let got = split_inkling(
            "<|message_model|><|content_thinking|>think<|end_message|>\
             <|message_model|><|content_text|>answer<|end_message|>\
             <|content_model_end_sampling|>",
            true,
        );
        assert_eq!(got, ("think".to_string(), "answer".to_string()));
    }

    /// No role tag: the bare channel markers are enough to start the walk.
    #[test]
    fn bare_channel_markers_split_without_the_role_tag() {
        assert_eq!(
            split_inkling("<|content_text|>hi<|end_message|>", true),
            (String::new(), "hi".to_string())
        );
    }

    /// A reply with no reasoning turn at all is content only.
    #[test]
    fn a_text_only_generation_has_no_reasoning() {
        assert_eq!(
            split_inkling(
                "<|message_model|><|content_text|>answer<|end_message|>",
                true
            ),
            (String::new(), "answer".to_string())
        );
    }

    /// A prefix that ends mid-marker holds the fragment back, so no half-tag reaches the client.
    #[test]
    fn a_streamed_prefix_never_leaks_a_partial_marker() {
        let (reasoning, content) =
            split_inkling("<|message_model|><|content_text|>Hi<|end_me", false);
        assert_eq!(reasoning, "");
        assert_eq!(content, "Hi");
        // And a prefix whose marker NAME is itself incomplete.
        let (reasoning, content) = split_inkling("<|message_model|><|content_te", false);
        assert_eq!((reasoning.as_str(), content.as_str()), ("", ""));
    }

    /// Ordinary streaming text arrives as it is produced — no hold once it cannot be a marker.
    #[test]
    fn a_streamed_prefix_of_plain_content_is_emitted() {
        assert_eq!(
            split_inkling("<|message_model|><|content_text|>Hel", false),
            (String::new(), "Hel".to_string())
        );
        assert_eq!(
            split_inkling("The sky is blue.<|", false),
            (String::new(), "The sky is blue.".to_string())
        );
    }

    /// A complete generation with no framing at all is the whole reply; a prefix of a marker is
    /// not content yet.
    #[test]
    fn an_unframed_generation_is_content_but_a_marker_prefix_is_not() {
        assert_eq!(
            split_inkling("plain answer", true),
            (String::new(), "plain answer".to_string())
        );
        assert_eq!(
            split_inkling("<|message_mod", false),
            (String::new(), String::new())
        );
        assert_eq!(
            split_inkling("<|message_model|>", true),
            (String::new(), String::new())
        );
    }

    /// An echoing model can open with ANY role tag; every one is stripped, so no raw marker
    /// reaches the client even when no channel tag follows it.
    #[test]
    fn a_bare_role_marker_is_stripped_from_unstructured_text() {
        for marker in ["<|message_system|>", "<|message_user|>", "<|message_tool|>"] {
            let echoed = format!("{marker}echo");
            assert_eq!(
                split_inkling(&echoed, true),
                (String::new(), "echo".to_string()),
                "{marker} leaked into the reply"
            );
        }
        // The model tag keeps working exactly as before.
        assert_eq!(
            split_inkling("<|message_model|>echo", true),
            (String::new(), "echo".to_string())
        );
    }

    /// The `developer` rewrite reaches the model as a `system` turn and is not dropped — the
    /// template refuses an unknown role outright.
    #[test]
    fn a_developer_turn_is_framed_as_a_system_turn() {
        let body = json!({"messages": [
            message("developer", "SENTINEL-DEV"),
            message("user", "hi")
        ]});
        let got = inkling_prompt(&body, (true, None)).unwrap();
        assert!(
            got.contains("<|message_system|><|content_text|>SENTINEL-DEV<|end_message|>"),
            "the developer turn did not reach the prompt as a system turn:\n{got}"
        );
        // The effort block lands BEFORE the user turn, not before the system one.
        let effort_at = got
            .find("Thinking effort level: 0.9")
            .expect("effort block");
        let user_at = got.find("<|message_user|>").expect("user turn");
        assert!(
            effort_at < user_at,
            "effort block after the user turn:\n{got}"
        );
    }

    /// `think == false` outranks an explicit effort, so a request that says "do not think" gets
    /// the zero level whatever strength word it also carried; with thinking on the word survives,
    /// and with no word at all the template's default lands.
    #[test]
    fn think_false_overrides_an_explicit_effort() {
        let body = json!({"messages": [message("user", "hi")]});
        for (think, effort, want) in [
            (
                false,
                Some("high"),
                "Thinking effort level: 0<|end_message|>",
            ),
            (
                true,
                Some("high"),
                "Thinking effort level: 0.9<|end_message|>",
            ),
            (true, None, "Thinking effort level: 0.9<|end_message|>"),
        ] {
            let got = inkling_prompt(&body, (think, effort)).unwrap();
            assert!(
                got.contains(want),
                "think={think} effort={effort:?}:\n{got}"
            );
        }
    }

    #[test]
    fn a_body_without_messages_is_refused() {
        let err = inkling_prompt(&json!({"prompt": "hi"}), (true, None)).unwrap_err();
        assert!(
            err.to_string().contains("`messages` array"),
            "unexpected refusal: {err}"
        );
    }
}
