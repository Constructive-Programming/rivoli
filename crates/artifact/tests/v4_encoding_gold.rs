//! `v4_encoding` against the four gold vectors the DeepSeek-V4-Flash checkpoint ships in
//! `encoding/tests/`, byte-for-byte.
//!
//! **This is the only test in the tree that can catch a wrongly-framed V4 prompt.** Nothing
//! downstream can: a prompt with the wrong turn markers still produces fluent text, and this
//! repo already lost months to exactly that on GLM's side (`encode_chat_turns` drifted onto
//! GLM-4's `<|role|>\n` framing and every benchmark before 2026-08-01 was measured off
//! template). The checkpoint's own `test_encoding_dsv4.py` is the executable specification;
//! this file is that specification, run against the Rust port.
//!
//! The gold is read from the checkpoint rather than copied into `tests/`, deliberately: a
//! vendored copy can be edited to make a failing port pass, and the whole value here is that
//! it cannot be. The unit tests inside `crates/artifact/src/v4_encoding/tests/` cover the paths
//! these four vectors miss and are hermetic, so a machine without the checkpoint is not left
//! with nothing.
//!
//! > **PORT NOTE 2026-08-16.** The reference splits this file's `RIVOLI_V4_SRC` lookup into
//! > `tests/common/f4_artifact_dir.rs`, shared with four `.f4` loading suites that do not exist
//! > in this tree yet. It is inlined here as [`checkpoint`] rather than added to
//! > `crates/artifact/tests/` with one caller, and the RULE it carries travelled with it: an
//! > explicitly-set env var that does not resolve is a FAILURE, not a skip. Move it out when
//! > the loading suites arrive; a shared helper with one caller is the shape that rots.
//!
//! Host-only: two JSON files and a 19 MB vocab, no device.

#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

use rivoli_artifact::tokenizer::Tokenizer;
use rivoli_artifact::v4_encoding::{
    EncodeOpts, ParsedToolCall, ThinkingMode, encode_messages, messages_from_openai,
    parse_message_from_completion_text,
};
use serde_json::Value;

/// The env var that points this suite at a checkpoint, and the path it defaults to.
const VAR: &str = "RIVOLI_V4_SRC";
const DEFAULT: &str = "/var/db/rivoli/deepseek-v4-flash-0731";

/// The checkpoint directory, or `None` when this machine has none.
///
/// **An explicitly-set `RIVOLI_V4_SRC` that does not resolve is a failure, not a skip.**
/// libtest captures stderr on passing tests, so an `eprintln!` skip is invisible in a green
/// run: someone who pointed this at a checkpoint and got all-pass would have no way to tell the
/// ground-truth cases never ran. The default path still skips, because a machine without the
/// 146 GB checkpoint is the ordinary case.
///
/// `probe` is a file that must exist inside it — the caller names what it actually needs — so a
/// directory holding half a checkpoint fails on the half that is missing rather than later and
/// elsewhere.
fn checkpoint(probe: &str) -> Option<String> {
    let named = std::env::var(VAR).ok();
    let dir = named.clone().unwrap_or_else(|| DEFAULT.into());
    if std::fs::metadata(format!("{dir}/{probe}")).is_ok() {
        return Some(dir);
    }
    assert!(
        named.is_none(),
        "{VAR}={dir} has no {probe} — refusing to pass by skipping"
    );
    eprintln!("SKIP: no checkpoint at {dir} (set {VAR})");
    None
}

/// The checkpoint's `encoding/tests/` folder — the four `(input, output)` pairs, located once.
///
/// A type rather than four functions that each take a `dir: &str`: the directory is one fact,
/// threading it through every helper made three quarters of this module's arguments strings, and
/// the code-health gate scores exactly that ratio.
struct Vectors {
    dir: String,
}

impl Vectors {
    /// The checkpoint's `encoding/` folder, or `None` when this machine has no checkpoint. The
    /// probe is the script this module ports, so a checkpoint missing it fails on the half that
    /// is missing rather than later and elsewhere.
    fn open() -> Option<Self> {
        checkpoint("encoding/encoding_dsv4.py").map(|dir| Self {
            dir: format!("{dir}/encoding"),
        })
    }

    fn read(&self, name: &str) -> String {
        let path = format!("{}/tests/{name}", self.dir);
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"))
    }

    fn gold(&self, n: u8) -> String {
        self.read(&format!("test_output_{n}.txt"))
    }

    fn input(&self, n: u8) -> Value {
        serde_json::from_str(&self.read(&format!("test_input_{n}.json"))).unwrap()
    }

    /// Encode the messages of `test_input_{n}.json` and diff against `test_output_{n}.txt`.
    fn check(&self, n: u8, messages: &[Value], thinking: ThinkingMode) -> String {
        let msgs = messages_from_openai(messages).unwrap();
        let got = encode_messages(msgs, &EncodeOpts::new(thinking)).unwrap();
        assert_eq!(got, self.gold(n), "test_output_{n}.txt mismatch");
        got
    }
}

/// Every completion-shaped span of `prompt`: after a thinking-mode assistant prefix, up to
/// the next user turn or the end.
fn assistant_turns(prompt: &str) -> Vec<&str> {
    const MARKER: &str = "<｜Assistant｜><think>";
    prompt
        .split(MARKER)
        .skip(1)
        .map(|turn| turn.split("<｜User｜>").next().unwrap_or(turn))
        .collect()
}

/// Case 1 — thinking mode with tools: tool schemas in the system turn, a DSML tool call, a
/// tool result merged into a following user turn. `messages[0]["tools"] = tools` is the
/// reference test's own setup, not ours.
#[test]
fn case_1_thinking_with_tools() {
    let Some(v) = Vectors::open() else { return };
    let td = v.input(1);
    let mut messages = td["messages"].as_array().unwrap().clone();
    messages[0]["tools"] = td["tools"].clone();
    let prompt = v.check(1, &messages, ThinkingMode::Thinking);

    // The reference's own round-trip: slice the two assistant turns back out of the prompt
    // it just built and parse them. This is what proves encode and parse are inverse — a
    // parser tested only on hand-written strings can agree with a wrong encoder.
    let turns = assistant_turns(&prompt);
    assert_eq!(turns.len(), 2, "{prompt:?}");
    let tc = parse_message_from_completion_text(turns[0], ThinkingMode::Thinking).unwrap();
    assert_eq!(
        tc.reasoning_content,
        "The user wants to know the weather in Beijing. I should use the get_weather tool."
    );
    assert_eq!(tc.content, "");
    assert_eq!(
        tc.tool_calls,
        vec![ParsedToolCall {
            name: "get_weather".into(),
            arguments: r#"{"location": "Beijing", "unit": "celsius"}"#.into(),
        }]
    );

    let fin = parse_message_from_completion_text(turns[1], ThinkingMode::Thinking).unwrap();
    assert_eq!(
        fin.reasoning_content,
        "Got the weather data. Let me format a nice response."
    );
    assert!(fin.content.contains("22°C"), "{:?}", fin.content);
    assert!(fin.tool_calls.is_empty());
}

/// Case 2 — thinking without tools: `drop_thinking` strips the earlier turn's reasoning.
#[test]
fn case_2_thinking_without_tools() {
    let Some(v) = Vectors::open() else { return };
    let messages = v.input(2);
    let prompt = v.check(2, messages.as_array().unwrap(), ThinkingMode::Thinking);

    let turns = assistant_turns(&prompt);
    let parsed =
        parse_message_from_completion_text(turns[turns.len() - 1], ThinkingMode::Thinking).unwrap();
    assert_eq!(
        parsed.reasoning_content,
        "The user asks about the capital of France. It is Paris."
    );
    assert_eq!(parsed.content, "The capital of France is Paris.");
    assert!(parsed.tool_calls.is_empty());
    // The load-bearing half of `drop_thinking`: the FIRST turn's reasoning is gone.
    assert!(!prompt.contains("The user said hello"));
}

/// Case 3 — interleaved thinking + search: a `developer` turn carrying tools, a
/// `latest_reminder`, and CJK content that must survive `ensure_ascii=False` unescaped.
#[test]
fn case_3_developer_tools_and_reminder() {
    let Some(v) = Vectors::open() else { return };
    let messages = v.input(3);
    v.check(3, messages.as_array().unwrap(), ThinkingMode::Thinking);
}

/// Case 4 — chat mode with a quick-instruction `action` task and a `latest_reminder`.
#[test]
fn case_4_quick_instruction_task() {
    let Some(v) = Vectors::open() else { return };
    let messages = v.input(4);
    v.check(4, messages.as_array().unwrap(), ThinkingMode::Chat);
}

/// Every framing token must tokenize to ONE id.
///
/// `v4_encoding` produces a STRING, unlike GLM's `encode_chat_turns` which assembles ids — so
/// the whole port rests on the `tokenizers` crate splitting on `added_tokens` inside ordinary
/// text. If it did not, `<｜User｜>` would encode as a handful of byte-BPE pieces the model has
/// never seen in that position, the prompt would be off template, and the output would still be
/// fluent. That is precisely the silent failure this module exists to prevent, so it gets its
/// own gate rather than an assumption in a comment.
///
/// Host-only: `Tokenizer::load` reads two JSON files and touches no device.
#[test]
fn special_tokens_survive_the_tokenizer() {
    // The CHECKPOINT's own tokenizer, not a converted artifact's. `convert_v4` copies
    // `tokenizer.json` and `generation_config.json` verbatim, so the two are the same bytes —
    // and reading the source means this gate does not need a 10 GB artifact to have been built.
    let Some(dir) = checkpoint("tokenizer.json") else {
        return;
    };
    let tok = Tokenizer::load(&dir).unwrap();

    // Ids from the checkpoint's own `tokenizer.json` `added_tokens`, read 2026-08-05.
    for (text, id) in [
        ("<｜begin▁of▁sentence｜>", 0_u32),
        ("<｜end▁of▁sentence｜>", 1),
        ("<｜User｜>", 128803),
        ("<｜Assistant｜>", 128804),
        ("<think>", 128821),
        ("</think>", 128822),
        ("｜DSML｜", 128825),
        ("<｜latest_reminder｜>", 128828),
        ("<｜action｜>", 128829),
        ("<｜title｜>", 128836),
    ] {
        assert_eq!(tok.encode(text).unwrap(), vec![id], "{text} alone");
    }

    // …and in context, which is the case that actually matters: a token glued to prose on
    // both sides must still come out whole. Filtering rather than `contains` so ORDER and
    // multiplicity are pinned too — a tokenizer that emitted `<｜User｜>` twice, or after
    // `<｜Assistant｜>`, would satisfy a containment check and still be off template.
    let ids = tok
        .encode("<｜begin▁of▁sentence｜>S<｜User｜>hi<｜Assistant｜><think>")
        .unwrap();
    let framing = [0_u32, 128803, 128804, 128821];
    let got: Vec<u32> = ids
        .iter()
        .copied()
        .filter(|i| framing.contains(i))
        .collect();
    assert_eq!(got, framing, "{ids:?}");
    // `<tool_result>` is deliberately NOT an added token — the reference emits it from a
    // plain string template, so it must tokenize as several ordinary pieces.
    assert!(tok.encode("<tool_result>").unwrap().len() > 1);
}
