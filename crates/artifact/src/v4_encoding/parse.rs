//! The inverse: one assistant completion's raw text back into reasoning, content and tool
//! calls.
//!
//! Split out of the reference's single 2822-line module (`old:src/artifact/dsv4_encoding.rs`)
//! by cohesion, under the 800-line cap. It shares only the special tokens with
//! [`super::render`] — deliberately, because they are the contract the round trip rests on,
//! and `super::tests::encode::encode_then_parse_is_the_identity_on_arguments` is what proves
//! the two agree about them.

use super::{BOS, DSML, EOS, THINK_CLOSE, THINK_OPEN, ThinkingMode};
use anyhow::{Result, bail, ensure};
use serde_json::Value;

/// One assistant turn, recovered from the model's raw completion text.
///
/// The reference's `"role": "assistant"` key is this type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedMessage {
    pub content: String,
    /// Empty in [`ThinkingMode::Chat`], where nothing is read before the content.
    pub reasoning_content: String,
    pub tool_calls: Vec<ParsedToolCall>,
}

/// A recovered tool call. `arguments` is a JSON string, as OpenAI wants it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedToolCall {
    pub name: String,
    pub arguments: String,
}

/// The UNREAD tail of a completion — every read below advances it.
///
/// A cursor rather than a `rest: &str` threaded through six functions. The reference passes the
/// remaining text as an argument and returns the new remainder beside every result, which here
/// made most of this module's arguments strings — the ratio the code-health gate scores — and,
/// more usefully, made "did this call advance?" a per-call-site reading rather than a property.
/// Every use in the reference is a sequential advancing read, so nothing is lost.
struct Scan<'t> {
    rest: &'t str,
}

impl<'t> Scan<'t> {
    const fn new(text: &'t str) -> Self {
        Self { rest: text }
    }

    fn is_empty(&self) -> bool {
        self.rest.is_empty()
    }

    /// Read up to the earliest of `stops`, advancing past the token that matched. Returns the
    /// text before it and which stop it was, or `(everything, None)` with the cursor emptied.
    ///
    /// Ties go to the earliest entry in `stops`, matching the reference's strict `pos < min_pos`
    /// against a list scanned in order.
    fn read_until(&mut self, stops: &[&str]) -> (&'t str, Option<usize>) {
        let hit = stops
            .iter()
            .enumerate()
            .filter_map(|(i, s)| self.rest.find(s).map(|p| (p, i, s.len())))
            .min_by_key(|&(p, i, _)| (p, i));
        // Both slices are on a `find` boundary plus a whole needle, so they are on char
        // boundaries by construction; `get` rather than `[..]` anyway, because the workspace
        // lint table does not deny `indexing_slicing` and a panic in a completion parser is a
        // crash on untrusted model output. An unreachable `None` degrades to "no stop found",
        // which every caller already handles as malformed.
        let split = hit.and_then(|(p, i, len)| {
            let before = self.rest.get(..p)?;
            let after = self.rest.get(p + len..)?;
            Some((before, after, i))
        });
        match split {
            Some((before, after, i)) => {
                self.rest = after;
                (before, Some(i))
            }
            None => {
                let all = self.rest;
                self.rest = "";
                (all, None)
            }
        }
    }
}

/// Python's `$`: matches at the end of the string, **or just before a single trailing
/// newline**. Both DSML regexes in the reference are anchored with it, so a value that ends
/// in `\n` is accepted there and must be here.
fn strip_anchored_suffix<'a>(s: &'a str, suffix: &str) -> Option<&'a str> {
    s.strip_suffix(suffix)
        .or_else(|| s.strip_suffix('\n').and_then(|t| t.strip_suffix(suffix)))
}

/// `to_json` of a string — the escaping the reference's `decode_dsml_to_arguments` applies
/// to keys and to `string="true"` values.
fn json_quote(s: &str) -> String {
    Value::String(s.to_string()).to_string()
}

/// `^\s*name="(.*?)">\n$` — the tool name between `<｜DSML｜invoke` and its first child.
fn parse_invoke_name(content: &str) -> Result<&str> {
    strip_anchored_suffix(content, "\">\n")
        .and_then(|s| s.trim_start().strip_prefix("name=\""))
        .ok_or_else(|| anyhow::anyhow!("tool name format error: '{content}'"))
}

/// `^ name="(.*?)" string="(true|false)">(.*?)<$` — one DSML parameter.
///
/// Hand-rolled rather than via `regex`, which is not a dependency of this crate. The two
/// `.*?` groups are non-greedy but the pattern is anchored at both ends, so: the NAME ends
/// at the *first* `" string="true|false">` and the VALUE runs to the *last* `<`. That is
/// what the scan below does, and it is why a value containing `<` round-trips.
fn parse_parameter(content: &str) -> Result<(&str, &str, bool)> {
    let err = || anyhow::anyhow!("parameter format error: '{content}'");
    let body = strip_anchored_suffix(content, "<").ok_or_else(err)?;
    let rest = body.strip_prefix(" name=\"").ok_or_else(err)?;
    for (i, _) in rest.char_indices() {
        // `get(i..)` rather than `[i..]`: `i` is a `char_indices` boundary so it cannot fail,
        // and the same no-panic rule applies as in `read_until_stop`.
        let Some(tail) = rest.get(i..) else { continue };
        for (mid, is_str) in [
            ("\" string=\"true\">", true),
            ("\" string=\"false\">", false),
        ] {
            if let Some(value) = tail.strip_prefix(mid) {
                let name = rest.get(..i).ok_or_else(err)?;
                return Ok((name, value, is_str));
            }
        }
    }
    Err(err())
}

/// The DSML tag spellings, resolved once per parse instead of at every `read_until_stop`.
///
/// A struct rather than five locals threaded through two functions: [`parse_tool_calls`] was
/// the reference's one body and splitting it (for the code-health gate) would otherwise have
/// meant a five-`&str` parameter list — a Primitive Obsession row and a call site where a
/// transposition type-checks, since all five are `String`.
struct Tags {
    invoke_open: String,
    invoke_close: String,
    param_open: String,
    param_close: String,
    calls_close: String,
}

impl Tags {
    fn new() -> Self {
        Self {
            invoke_open: format!("<{DSML}invoke"),
            invoke_close: format!("</{DSML}invoke"),
            param_open: format!("<{DSML}parameter"),
            param_close: format!("/{DSML}parameter"),
            calls_close: format!("</{DSML}tool_calls>"),
        }
    }

    /// One `<｜DSML｜invoke>`'s parameters, read off `scan`, as the reconstructed `arguments`
    /// object. `stop` on entry is which token ended the NAME read — `Some(0)` means a parameter
    /// follows.
    ///
    /// A method rather than a free function taking `&Tags`: [`Self::parse_tool_calls`] below is
    /// the other half of the same split, and both threading the tag table as an argument made
    /// most of this module's arguments strings.
    fn parse_parameters(&self, scan: &mut Scan<'_>, mut stop: Option<usize>) -> Result<String> {
        // Keys in insertion order, so the reconstructed `arguments` matches the order the
        // model emitted. A Vec rather than a Map because the duplicate check is explicit.
        let mut args: Vec<(String, String)> = Vec::new();
        while stop == Some(0) {
            let (param, _) = scan.read_until(&[&self.param_close]);
            let (key, value, is_str) = parse_parameter(param)?;
            ensure!(
                !args.iter().any(|(k, _)| k == key),
                "duplicate parameter name: '{key}'"
            );
            let rendered = if is_str {
                json_quote(value)
            } else {
                value.to_string()
            };
            args.push((key.to_string(), rendered));

            let (gap, next) = scan.read_until(&[&self.param_open, &self.invoke_close]);
            ensure!(
                gap == ">\n",
                "parameter format error: expected '>\\n' got '{gap}'"
            );
            stop = next;
        }
        let body = args
            .iter()
            .map(|(k, v)| format!("{}: {v}", json_quote(k)))
            .collect::<Vec<_>>()
            .join(", ");
        Ok(format!("{{{body}}}"))
    }

    /// Parse a `<｜DSML｜tool_calls>` block off `scan`, starting just after its opening tag.
    ///
    /// The reference's loop condition is `while index < len(text)`, so a truncated block yields
    /// the calls parsed so far rather than an error — deliberate, and reproduced.
    fn parse_tool_calls(&self, scan: &mut Scan<'_>) -> Result<Vec<ParsedToolCall>> {
        let mut calls = Vec::new();
        while !scan.is_empty() {
            let (gap, stop) = scan.read_until(&[&self.invoke_open, &self.calls_close]);
            ensure!(
                gap == ">\n",
                "tool call format error: expected '>\\n' got '{gap}'"
            );
            // Positional: 0 is `invoke_open`, 1 is `calls_close`, in the slice above. Spelled
            // out rather than `_ => {}` so reordering that slice cannot silently change which
            // token continues the loop.
            match stop {
                Some(0) => {}
                Some(1) => break,
                // NOT `unreachable!`: `Scan::read_until` returns an index into the two-entry
                // slice it was given, so this arm is dead — but a panic macro on an
                // untrusted-input path is a crash a refusal would have handled, and the
                // reference's own note is that this parser refuses malformed output rather
                // than repairing it.
                Some(other) => bail!("read_until returned stop {other} for a 2-entry slice"),
                None => bail!("missing special token in tool calls"),
            }

            let (name_content, stop) = scan.read_until(&[&self.param_open, &self.invoke_close]);
            let name = parse_invoke_name(name_content)?.to_string();
            let arguments = self.parse_parameters(scan, stop)?;
            calls.push(ParsedToolCall { name, arguments });
        }
        Ok(calls)
    }
}

/// Parse one assistant completion into reasoning, content and tool calls.
///
/// > **The reference's own caveat, which applies verbatim to this port:** it "is designed to
/// > handle well-formatted model output only. It does not attempt to correct or recover from
/// > malformed output that the model might occasionally generate."
///
/// Every `assert` in the reference is an `Err` here rather than a panic — the model is an
/// untrusted producer of this string, and a decode loop must be able to say "that came back
/// malformed" without taking the process down.
pub fn parse_message_from_completion_text(
    text: &str,
    thinking: ThinkingMode,
) -> Result<ParsedMessage> {
    let tool_calls_open = format!("\n\n<{DSML}tool_calls");
    let mut scan = Scan::new(text);

    let reasoning_content = match thinking {
        ThinkingMode::Chat => String::new(),
        ThinkingMode::Thinking => {
            let (reasoning, stop) = scan.read_until(&[THINK_CLOSE, &tool_calls_open]);
            ensure!(stop == Some(0), "invalid thinking format: missing </think>");
            reasoning.to_string()
        }
    };

    let (content, stop) = scan.read_until(&[EOS, &tool_calls_open]);
    let (content, is_tool_calling) = match stop {
        Some(1) => (content.to_string(), true),
        Some(0) => (content.to_string(), false),
        _ => bail!("invalid format: missing EOS token"),
    };

    let tool_calls = if is_tool_calling {
        let calls = Tags::new().parse_tool_calls(&mut scan)?;
        let (tail, _) = scan.read_until(&[EOS]);
        ensure!(tail.is_empty(), "unexpected content after tool calls");
        calls
    } else {
        Vec::new()
    };
    ensure!(scan.is_empty(), "unexpected content at end");

    let parsed = ParsedMessage {
        content,
        reasoning_content,
        tool_calls,
    };
    parsed.ensure_no_control_tokens()?;
    Ok(parsed)
}

impl ParsedMessage {
    /// The model must not have written a control token into prose: if it did, the framing of
    /// everything after it is a guess, and a guess is what silently corrupts a transcript.
    ///
    /// A method on the finished message rather than a function over its two text fields —
    /// which is also the shape that keeps this module's argument list from being almost all
    /// strings, the ratio the code-health gate scores.
    fn ensure_no_control_tokens(&self) -> Result<()> {
        for token in [BOS, EOS, THINK_OPEN, THINK_CLOSE, DSML] {
            ensure!(
                !self.content.contains(token) && !self.reasoning_content.contains(token),
                "unexpected special token '{token}' in content"
            );
        }
        Ok(())
    }
}
