//! The PARSER: model output back into reasoning, content and tool calls — and the refusals
//! that keep a malformed completion from being silently repaired.
//!
//! See [`super`] for how the expectations were produced and why this module lives in `src/`.

use super::*;

/// A DSML tool-call block, assembled the way the model emits it.
///
/// Here rather than in [`super`] because this is its only consumer — see the note there.
fn dsml(params: &str, name: &str) -> String {
    format!(
        "\n\n<{DSML}tool_calls>\n<{DSML}invoke name=\"{name}\">\n{params}</{DSML}invoke>\n</{DSML}tool_calls>"
    )
}

fn param(name: &str, is_str: bool, value: &str) -> String {
    format!("<{DSML}parameter name=\"{name}\" string=\"{is_str}\">{value}</{DSML}parameter>\n")
}

fn refused(text: &str, mode: ThinkingMode) -> String {
    parse_message_from_completion_text(text, mode)
        .unwrap_err()
        .to_string()
}

#[test]
fn parse_reasoning_and_content() {
    // The README's own round-trip example.
    let p = parsed(
        &format!("Simple arithmetic.</think>2 + 2 = 4.{EOS}"),
        ThinkingMode::Thinking,
    );
    assert_eq!(p.reasoning_content, "Simple arithmetic.");
    assert_eq!(p.content, "2 + 2 = 4.");
    assert!(p.tool_calls.is_empty());

    // Chat mode reads no reasoning block at all.
    let c = parsed(&format!("2 + 2 = 4.{EOS}"), ThinkingMode::Chat);
    assert_eq!(c.reasoning_content, "");
    assert_eq!(c.content, "2 + 2 = 4.");

    assert_eq!(
        parsed(EOS, ThinkingMode::Chat),
        ParsedMessage {
            content: String::new(),
            reasoning_content: String::new(),
            tool_calls: vec![]
        }
    );
}

#[test]
fn parse_tool_calls_inverts_the_encoder() {
    let call = |body: &str| {
        parsed(
            &format!("R</think>{}{EOS}", dsml(body, "f")),
            ThinkingMode::Thinking,
        )
    };

    // `string="true"` is JSON-quoted back; `string="false"` is spliced in verbatim, so
    // whatever JSON the model wrote survives byte-for-byte.
    assert_eq!(
        call(&param("k", true, "v")).tool_calls,
        vec![ParsedToolCall {
            name: "f".into(),
            arguments: r#"{"k": "v"}"#.into()
        }]
    );
    let mixed = format!("{}{}", param("n", false, "5"), param("a", false, "[1, 2]"));
    assert_eq!(
        call(&mixed).tool_calls[0].arguments,
        r#"{"n": 5, "a": [1, 2]}"#
    );
    assert_eq!(call("").tool_calls[0].arguments, "{}");

    // A value that contains `<`, a `"`, a `\` or a newline still round-trips: the
    // reference's regex is anchored at the end, so the value runs to the LAST `<`.
    assert_eq!(
        call(&param("k", true, "a<b")).tool_calls[0].arguments,
        r#"{"k": "a<b"}"#
    );
    assert_eq!(
        call(&param("k", true, "a\"b\\c")).tool_calls[0].arguments,
        r#"{"k": "a\"b\\c"}"#
    );
    assert_eq!(
        call(&param("k", true, "line1\nline2")).tool_calls[0].arguments,
        r#"{"k": "line1\nline2"}"#
    );
    // Python's `$` also matches before ONE trailing newline, so a value ending in `\n`
    // parses rather than erroring.
    assert_eq!(
        call(&param("k", true, "v\n")).tool_calls[0].arguments,
        r#"{"k": "v\n"}"#
    );

    // Prose before the block is content; the `\n\n` that opens the block is not.
    let with_prose = parsed(
        &format!("R</think>Let me look.{}{EOS}", dsml("", "f")),
        ThinkingMode::Thinking,
    );
    assert_eq!(with_prose.content, "Let me look.");

    let two = format!(
        "\n\n<{DSML}tool_calls>\n<{DSML}invoke name=\"f\">\n{}</{DSML}invoke>\n<{DSML}invoke name=\"g\">\n</{DSML}invoke>\n</{DSML}tool_calls>",
        param("k", true, "v")
    );
    let both = parsed(&format!("R</think>{two}{EOS}"), ThinkingMode::Thinking);
    let names: Vec<&str> = both.tool_calls.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(names, ["f", "g"]);

    // A truncated block yields what it parsed rather than an error — the reference's
    // loop condition is `index < len(text)`, so it just runs out.
    let cut = parsed(
        &format!("R</think>\n\n<{DSML}tool_calls>\n<{DSML}invoke name=\"f\">\n"),
        ThinkingMode::Thinking,
    );
    assert_eq!(cut.tool_calls[0].arguments, "{}");

    assert_eq!(
        parsed(&format!("prose{}{EOS}", dsml("", "f")), ThinkingMode::Chat).content,
        "prose"
    );
}

/// Malformed output is REFUSED, not repaired. The reference's own note: it "is designed
/// to handle well-formatted model output only."
#[test]
fn parse_refuses_malformed_output() {
    assert!(refused("R</think>hello", ThinkingMode::Thinking).contains("missing EOS"));
    assert!(refused("", ThinkingMode::Chat).contains("missing EOS"));
    assert!(refused(&format!("hello{EOS}"), ThinkingMode::Thinking).contains("missing </think>"));
    // Tool calls before `</think>` in thinking mode: the reasoning read stops on the
    // block and reports the missing close rather than silently treating it as reasoning.
    assert!(
        refused(&format!("{}{EOS}", dsml("", "f")), ThinkingMode::Thinking)
            .contains("missing </think>")
    );
    assert!(
        refused(&format!("R</think>a{EOS}b{EOS}"), ThinkingMode::Thinking)
            .contains("unexpected content at end")
    );
    assert!(
        refused(
            &format!("R</think>{}tail{EOS}", dsml("", "f")),
            ThinkingMode::Thinking
        )
        .contains("after tool calls")
    );
    let dup = format!("{}{}", param("k", true, "1"), param("k", true, "2"));
    assert!(
        refused(
            &format!("R</think>{}{EOS}", dsml(&dup, "f")),
            ThinkingMode::Thinking
        )
        .contains("duplicate parameter")
    );

    // A control token inside prose means the framing of everything after it is a guess.
    assert!(
        refused(
            &format!("R</think>he<think>llo{EOS}"),
            ThinkingMode::Thinking
        )
        .contains("special token '<think>'")
    );
    assert!(
        refused(&format!("R</think>a{DSML}b{EOS}"), ThinkingMode::Thinking)
            .contains("special token '｜DSML｜'")
    );
}
