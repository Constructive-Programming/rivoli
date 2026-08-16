//! Every expectation under this module was produced by RUNNING the checkpoint's
//! `encoding/encoding_dsv4.py`, never by reading its README. To regenerate one:
//!
//! ```text
//! cd /var/db/rivoli/deepseek-v4-flash-0731/encoding
//! python3 -c 'from encoding_dsv4 import encode_messages as e; print(repr(e([...], thinking_mode="thinking")))'
//! ```
//!
//! Inputs go in as OpenAI-shaped JSON so they can be transcribed from a Python call
//! literally — which also puts [`messages_from_openai`] on every one of these paths.
//!
//! **INSIDE `src/`, not in `crates/artifact/tests/`, unlike its Glimmer twin.** Two of these
//! tests need items the crate does not export: `DSML` is a private const of [`super`] (the
//! header explains why it is the *bare* token, and making it `pub` to test it would widen the
//! API for a test's convenience), and `python_json` is `pub(crate)` in
//! [`crate::tokenizer`]. `crates/artifact/tests/v4_encoding_gold.rs` holds everything that
//! needs only the public surface — the four gold vectors and the tokenizer gate — and needs
//! the checkpoint on disk, where these are hermetic.
//!
//! Three files under 800 lines rather than one of 1200, split by what they interrogate: the
//! ENCODER's bytes, the BOUNDARY's accept/reject edge, and the PARSER. The helpers below are
//! shared by all three, which is what makes this a module with submodules rather than three
//! sibling files — jscpd reports a copied `ok`/`refuses` pair on sight.

#![allow(clippy::unwrap_used, clippy::expect_used)]

mod boundary;
mod encode;
mod parse;

use super::*;
use anyhow::Result;
use serde_json::{Value, json};

/// `messages` in, prompt out. Mirrors `encode_messages(messages, thinking_mode=…)`.
fn enc(messages: &Value, opts: &EncodeOpts) -> Result<String> {
    encode_messages(messages_from_openai(messages.as_array().unwrap())?, opts)
}

fn ok(messages: &Value, opts: &EncodeOpts) -> String {
    enc(messages, opts).unwrap()
}

/// Shared by the refusal assertions below.
#[track_caller]
fn assert_refused(e: &anyhow::Error, needle: &str) {
    let e = e.to_string();
    assert!(
        e.contains(needle),
        "expected {needle:?} in the refusal, got {e:?}"
    );
}

/// Assert the BOUNDARY refuses `msg`, and say what it actually said when it does not.
/// `#[track_caller]` so the failure points at the case, not at this line.
#[track_caller]
fn refuses(msg: &Value, needle: &str) {
    assert_refused(&message_from_openai(msg).unwrap_err(), needle);
}

const HELPFUL: &str = "You are a helpful assistant.";

fn helpful_2plus2() -> Value {
    json!([{"role": "system", "content": HELPFUL}, {"role": "user", "content": "What is 2+2?"}])
}

/// `system, <the turn under test>, assistant(A/R), user(U2)` — the shape for asking what
/// happens to a turn that sits BEFORE the last user message, which is the only place
/// `drop_thinking` does anything. Callers vary the first two turns and nothing else.
fn before_last_user(system: Value, second: Value) -> Value {
    json!([
        system,
        second,
        {"role": "assistant", "content": "A", "reasoning_content": "R"},
        {"role": "user", "content": "U2"},
    ])
}

/// `system, user, assistant-calling-two-tools` — the head both the DSML render test and
/// the tool-result ordering test build on. `f` is asked for first, `g` second, and the
/// ids are what a later tool result sorts against.
fn two_call_conversation() -> Value {
    json!([
        {"role": "system", "content": "S"},
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "reasoning_content": "R", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{\"x\": \"1\"}"}},
            {"id": "c2", "type": "function", "function": {"name": "g", "arguments": "{\"y\": \"2\"}"}}]},
    ])
}

fn parsed(text: &str, mode: ThinkingMode) -> ParsedMessage {
    parse_message_from_completion_text(text, mode).unwrap()
}

// `dsml` and `param` — the two DSML block builders — live in `tests/parse.rs` rather than here,
// even though `encode.rs` is where the round-trip case uses `parsed`. They have exactly one
// consumer, and moving them there is what keeps THIS module under the code-health gate's
// string-argument ratio (five of its arguments were theirs).
