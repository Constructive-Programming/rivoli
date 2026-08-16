//! The ENCODER's bytes: what a conversation renders to, for every framing decision the
//! reference makes. See [`super`] for how the expectations were produced and why this module
//! lives in `src/`.

use super::*;

/// The README's quick-start line, the chat-mode twin, and the two reasoning-effort
/// prefixes — which are pure text prepended before the system message and are the one
/// part of this port with no gold vector in `encoding/tests/`.
#[test]
fn generation_prompt_and_effort_prefixes() {
    let m = helpful_2plus2();
    let thinking = EncodeOpts::new(ThinkingMode::Thinking);
    let chat = EncodeOpts::new(ThinkingMode::Chat);
    // A closure, not three `EncodeOpts` setter methods: rustfmt explodes an inline
    // `EncodeOpts { .. }` across six lines and three of those blocks are then a jscpd
    // clone.
    let effort = |base: EncodeOpts, reasoning_effort| EncodeOpts {
        reasoning_effort,
        ..base
    };

    assert_eq!(
        ok(&m, &thinking),
        "<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>What is 2+2?<｜Assistant｜><think>"
    );
    assert_eq!(
        ok(&m, &chat),
        "<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>What is 2+2?<｜Assistant｜></think>"
    );

    assert_eq!(
        ok(&m, &effort(thinking, ReasoningEffort::High)),
        "<｜begin▁of▁sentence｜>Reasoning Effort: Absolute maximum with no shortcuts permitted.\nYou MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\nExplicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\nYou are a helpful assistant.<｜User｜>What is 2+2?<｜Assistant｜><think>"
    );
    assert_eq!(
        ok(&m, &effort(thinking, ReasoningEffort::Max)),
        "<｜begin▁of▁sentence｜>Reasoning Effort: Beyond maximum — exhaustive, relentless, and uncompromising.\nYou MUST reason with the utmost depth and rigor, leaving absolutely nothing to chance: exhaustively decompose the problem into its most fundamental components, trace every causal chain to its root, and resolve the underlying cause rather than any surface symptom.\nDo not stop reasoning until you have independently verified the solution from multiple angles and are certain that no assumption remains unchecked and no error remains undiscovered.\n\nYou are a helpful assistant.<｜User｜>What is 2+2?<｜Assistant｜><think>"
    );

    // Effort is a THINKING-mode knob: chat mode has no reasoning block to lengthen, so
    // the prefix must not appear. Silently prepending it would be a prompt the model was
    // never trained on.
    assert_eq!(ok(&m, &effort(chat, ReasoningEffort::High)), ok(&m, &chat));

    assert_eq!(
        ok(
            &m,
            &EncodeOpts {
                add_bos: false,
                ..thinking
            }
        ),
        "You are a helpful assistant.<｜User｜>What is 2+2?<｜Assistant｜><think>"
    );
}

/// A missing or empty system turn contributes nothing at all — not a blank line, not a
/// separator. BOS then straight into `<｜User｜>`.
#[test]
fn system_turn_is_bare_text() {
    let thinking = EncodeOpts::new(ThinkingMode::Thinking);
    let no_system = json!([{"role": "user", "content": "What is 2+2?"}]);
    assert_eq!(
        ok(&no_system, &thinking),
        "<｜begin▁of▁sentence｜><｜User｜>What is 2+2?<｜Assistant｜><think>"
    );

    let empty =
        json!([{"role": "system", "content": ""}, {"role": "user", "content": "What is 2+2?"}]);
    assert_eq!(
        ok(&empty, &EncodeOpts::new(ThinkingMode::Chat)),
        "<｜begin▁of▁sentence｜><｜User｜>What is 2+2?<｜Assistant｜></think>"
    );
}

/// `drop_thinking` strips reasoning from assistant turns before the last user message —
/// and `<｜Assistant｜>` for those turns closes `</think>` immediately, because there is
/// no reasoning to open it for.
#[test]
fn drop_thinking_strips_earlier_reasoning() {
    let thinking = EncodeOpts::new(ThinkingMode::Thinking);
    let two_turns = json!([
        {"role": "system", "content": "S"},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "reasoning_content": "R1", "content": "Hello"},
        {"role": "user", "content": "Bye"},
    ]);
    assert_eq!(
        ok(&two_turns, &thinking),
        "<｜begin▁of▁sentence｜>S<｜User｜>Hi<｜Assistant｜></think>Hello<｜end▁of▁sentence｜><｜User｜>Bye<｜Assistant｜><think>"
    );
    // Turned off, every turn keeps its reasoning and every prefix opens `<think>`.
    assert_eq!(
        ok(
            &two_turns,
            &EncodeOpts {
                drop_thinking: false,
                ..thinking
            }
        ),
        "<｜begin▁of▁sentence｜>S<｜User｜>Hi<｜Assistant｜><think>R1</think>Hello<｜end▁of▁sentence｜><｜User｜>Bye<｜Assistant｜><think>"
    );

    // An assistant turn AFTER the last user is a continuation and keeps its reasoning.
    let trailing = json!([
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
        {"role": "assistant", "content": "A", "reasoning_content": "R"},
    ]);
    assert_eq!(
        ok(&trailing, &thinking),
        "<｜begin▁of▁sentence｜>S<｜User｜>U<｜Assistant｜><think>R</think>A<｜end▁of▁sentence｜>"
    );

    // An EMPTY `tools` list is falsy in Python and must NOT disable drop_thinking.
    let empty_tools = before_last_user(
        json!({"role": "system", "content": "S", "tools": []}),
        json!({"role": "user", "content": "U"}),
    );
    assert_eq!(
        ok(&empty_tools, &thinking),
        "<｜begin▁of▁sentence｜>S<｜User｜>U<｜Assistant｜></think>A<｜end▁of▁sentence｜><｜User｜>U2<｜Assistant｜><think>"
    );
}

/// A `developer` turn before the last user message is DROPPED ENTIRELY — content, tools
/// and its `<｜User｜>` prefix with it. So the assistant reply that followed it ends up
/// glued to the system turn with no assistant marker at all. Surprising, and the
/// reference's; reproduced rather than corrected.
#[test]
fn developer_turns_before_the_last_user_vanish() {
    let m = before_last_user(
        json!({"role": "system", "content": HELPFUL}),
        json!({"role": "developer", "content": "D"}),
    );
    assert_eq!(
        ok(&m, &EncodeOpts::new(ThinkingMode::Thinking)),
        "<｜begin▁of▁sentence｜>You are a helpful assistant.A<｜end▁of▁sentence｜><｜User｜>U2<｜Assistant｜><think>"
    );
}

/// String arguments go in raw with `string="true"`; every other JSON type goes in as
/// JSON with `string="false"`. That asymmetry is what makes the parse invertible.
#[test]
fn tool_calls_render_as_dsml() {
    let thinking = EncodeOpts::new(ThinkingMode::Thinking);
    let call = |args: &str| {
        json!([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "reasoning_content": "R", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "f", "arguments": args}}]},
        ])
    };

    let mixed = r#"{"s": "txt", "n": 5, "b": true, "a": [1, 2], "o": {"k": "v"}, "z": null}"#;
    assert_eq!(
        ok(&call(mixed), &thinking),
        "<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>go<｜Assistant｜><think>R</think>\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"f\">\n<｜DSML｜parameter name=\"s\" string=\"true\">txt</｜DSML｜parameter>\n<｜DSML｜parameter name=\"n\" string=\"false\">5</｜DSML｜parameter>\n<｜DSML｜parameter name=\"b\" string=\"false\">true</｜DSML｜parameter>\n<｜DSML｜parameter name=\"a\" string=\"false\">[1, 2]</｜DSML｜parameter>\n<｜DSML｜parameter name=\"o\" string=\"false\">{\"k\": \"v\"}</｜DSML｜parameter>\n<｜DSML｜parameter name=\"z\" string=\"false\">null</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls><｜end▁of▁sentence｜>"
    );

    // Valid JSON that is not an OBJECT: `json.loads` succeeds and `.items()` then raises
    // AttributeError. Refused rather than guessed at.
    assert_refused(&enc(&call("5"), &thinking).unwrap_err(), "not an object");

    assert_eq!(
        ok(&two_call_conversation(), &thinking),
        "<｜begin▁of▁sentence｜>S<｜User｜>go<｜Assistant｜><think>R</think>\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"f\">\n<｜DSML｜parameter name=\"x\" string=\"true\">1</｜DSML｜parameter>\n</｜DSML｜invoke>\n<｜DSML｜invoke name=\"g\">\n<｜DSML｜parameter name=\"y\" string=\"true\">2</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls><｜end▁of▁sentence｜>"
    );
}

/// Tool results become `<tool_result>` blocks inside a user turn, `\n\n`-joined, and are
/// reordered to match the order the assistant asked for them.
#[test]
fn tool_results_merge_into_the_user_turn() {
    let thinking = EncodeOpts::new(ThinkingMode::Thinking);
    let head = two_call_conversation();
    let with = |extra: Value| {
        let mut m = head.as_array().unwrap().clone();
        m.extend(extra.as_array().unwrap().iter().cloned());
        Value::Array(m)
    };

    // Results arrive c2-then-c1 and are emitted c1-then-c2.
    let out_of_order = with(json!([
        {"role": "tool", "tool_call_id": "c2", "content": "SECOND"},
        {"role": "tool", "tool_call_id": "c1", "content": "FIRST"},
    ]));
    assert!(ok(&out_of_order, &thinking).ends_with("<｜User｜><tool_result>FIRST</tool_result>\n\n<tool_result>SECOND</tool_result><｜Assistant｜><think>"), "{}", ok(&out_of_order, &thinking));

    // A user message after a tool result joins the SAME turn, `\n\n` between them.
    let then_text = with(json!([
        {"role": "tool", "tool_call_id": "c1", "content": "RES"},
        {"role": "user", "content": "and also this"},
    ]));
    assert!(
        ok(&then_text, &thinking).ends_with(
            "<｜User｜><tool_result>RES</tool_result>\n\nand also this<｜Assistant｜><think>"
        ),
        "{}",
        ok(&then_text, &thinking)
    );

    // Two plain user messages in a row are one turn.
    let two_users = json!([{"role": "system", "content": "S"}, {"role": "user", "content": "A"}, {"role": "user", "content": "B"}]);
    assert_eq!(
        ok(&two_users, &EncodeOpts::new(ThinkingMode::Chat)),
        "<｜begin▁of▁sentence｜>S<｜User｜>A\n\nB<｜Assistant｜></think>"
    );

    // A tool result with no preceding user turn opens one.
    let orphan = json!([{"role": "system", "content": "S"}, {"role": "tool", "tool_call_id": "c", "content": "R"}]);
    assert_eq!(
        ok(&orphan, &thinking),
        "<｜begin▁of▁sentence｜>S<｜User｜><tool_result>R</tool_result><｜Assistant｜><think>"
    );

    // An id-less call still CONSUMES an index in the reference's `enumerate`, so `c1`
    // here is key 1 and the unmatched result keeps the default 0 — unmatched first. A
    // port that filtered the id-less call out and used a POSITION would renumber `c1` to
    // 0 and emit them the other way round. Regression for a review finding, 2026-08-05,
    // and — with the differential gate unported (see this module's parent header) — one of
    // the two cases that now stand in for it.
    let no_id = json!([
        {"role": "system", "content": "S"},
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "f", "arguments": "{}"}},
            {"id": "c1", "type": "function", "function": {"name": "g", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "RES_C1"},
        {"role": "tool", "tool_call_id": "zz", "content": "RES_UNKNOWN"},
    ]);
    assert!(ok(&no_id, &thinking).ends_with("<｜User｜><tool_result>RES_UNKNOWN</tool_result>\n\n<tool_result>RES_C1</tool_result><｜Assistant｜><think>"), "{}", ok(&no_id, &thinking));

    // A repeated id keeps the LAST index (the reference assigns into a dict), so `d` is
    // key 2 and sorts after `c`. The second of the two cases standing in for the
    // differential.
    let dup_id = json!([
        {"role": "system", "content": "S"},
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "d", "type": "function", "function": {"name": "f", "arguments": "{}"}},
            {"id": "c", "type": "function", "function": {"name": "g", "arguments": "{}"}},
            {"id": "d", "type": "function", "function": {"name": "h", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "d", "content": "D"},
        {"role": "tool", "tool_call_id": "c", "content": "C"},
    ]);
    assert!(ok(&dup_id, &thinking).ends_with("<｜User｜><tool_result>C</tool_result>\n\n<tool_result>D</tool_result><｜Assistant｜><think>"), "{}", ok(&dup_id, &thinking));

    // A user turn that already carries a task still ABSORBS a tool result — the
    // reference checks the task only on the text-merging branch, not this one.
    let after_task = json!([
        {"role": "system", "content": "S"},
        {"role": "user", "content": "P", "task": "query"},
        {"role": "tool", "tool_call_id": "c", "content": "R"},
    ]);
    assert_eq!(
        ok(&after_task, &EncodeOpts::new(ThinkingMode::Chat)),
        "<｜begin▁of▁sentence｜>S<｜User｜>P\n\n<tool_result>R</tool_result><｜query｜>"
    );
}

/// Quick-instruction tasks. `action` alone goes AFTER the assistant prefix and its
/// thinking token; the rest are appended directly to the message they sit on.
#[test]
fn quick_instruction_tasks() {
    let chat = EncodeOpts::new(ThinkingMode::Chat);
    for (task, token) in [
        ("query", "<｜query｜>"),
        ("authority", "<｜authority｜>"),
        ("domain", "<｜domain｜>"),
        ("read_url", "<｜read_url｜>"),
    ] {
        let m = json!([{"role": "system", "content": "S"}, {"role": "user", "content": "P", "task": task}]);
        assert_eq!(
            ok(&m, &chat),
            format!("<｜begin▁of▁sentence｜>S<｜User｜>P{token}"),
            "task {task}"
        );
    }

    let action = json!([{"role": "system", "content": "S"}, {"role": "user", "content": "P", "task": "action"}]);
    assert_eq!(
        ok(&action, &EncodeOpts::new(ThinkingMode::Thinking)),
        "<｜begin▁of▁sentence｜>S<｜User｜>P<｜Assistant｜><think><｜action｜>"
    );
    assert_eq!(
        ok(&action, &chat),
        "<｜begin▁of▁sentence｜>S<｜User｜>P<｜Assistant｜></think><｜action｜>"
    );

    // `title` sits on an ASSISTANT turn, after its EOS. The turn that follows is the
    // generated title, and it carries no reasoning block because the turn before it had
    // a task.
    let title = json!([
        {"role": "system", "content": "S"},
        {"role": "user", "content": "P"},
        {"role": "assistant", "content": "A", "reasoning_content": "R", "task": "title"},
        {"role": "assistant", "content": "T"},
    ]);
    assert_eq!(
        ok(&title, &EncodeOpts::new(ThinkingMode::Thinking)),
        "<｜begin▁of▁sentence｜>S<｜User｜>P<｜Assistant｜><think>R</think>A<｜end▁of▁sentence｜><｜title｜>T<｜end▁of▁sentence｜>"
    );

    assert!(
        message_from_openai(&json!({"role": "user", "content": "x", "task": "nope"}))
            .unwrap_err()
            .to_string()
            .contains("invalid task")
    );
}

/// `response_format` renders a `## Response Format:` block after the tools; `wo_eos`
/// leaves the assistant turn unterminated so the model continues it.
#[test]
fn response_format_and_wo_eos() {
    let m = json!([{"role": "system", "content": "S", "response_format": {"type": "json_object"}}, {"role": "user", "content": "P"}]);
    assert_eq!(
        ok(&m, &EncodeOpts::new(ThinkingMode::Chat)),
        "<｜begin▁of▁sentence｜>S\n\n## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n{\"type\": \"json_object\"}<｜User｜>P<｜Assistant｜></think>"
    );

    // Falsy in Python, so no heading at all.
    let empty = json!([{"role": "system", "content": "S", "response_format": {}}, {"role": "user", "content": "P"}]);
    assert_eq!(
        ok(&empty, &EncodeOpts::new(ThinkingMode::Chat)),
        "<｜begin▁of▁sentence｜>S<｜User｜>P<｜Assistant｜></think>"
    );

    let partial = json!([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "P"},
        {"role": "assistant", "content": "partial", "reasoning_content": "R", "wo_eos": true},
    ]);
    assert_eq!(
        ok(&partial, &EncodeOpts::new(ThinkingMode::Thinking)),
        "<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>P<｜Assistant｜><think>R</think>partial"
    );
}

/// `latest_reminder` is bare `<｜latest_reminder｜>{content}`, and — this is the odd one —
/// a turn followed by a reminder still emits its transition tokens FIRST, so the
/// reminder lands after `<｜Assistant｜><think>`. Reproduced, not corrected: that is the
/// sequence the model was trained on.
#[test]
fn latest_reminder_placement() {
    let thinking = EncodeOpts::new(ThinkingMode::Thinking);
    let before_user = json!([{"role": "system", "content": "S"}, {"role": "latest_reminder", "content": "LR"}, {"role": "user", "content": "P"}]);
    assert_eq!(
        ok(&before_user, &thinking),
        "<｜begin▁of▁sentence｜>S<｜latest_reminder｜>LR<｜User｜>P<｜Assistant｜><think>"
    );

    let after_user = json!([
        {"role": "system", "content": "S"},
        {"role": "user", "content": "P"},
        {"role": "latest_reminder", "content": "LR"},
        {"role": "assistant", "content": "A", "reasoning_content": "R"},
    ]);
    assert_eq!(
        ok(&after_user, &thinking),
        "<｜begin▁of▁sentence｜>S<｜User｜>P<｜Assistant｜><think><｜latest_reminder｜>LRR</think>A<｜end▁of▁sentence｜>"
    );
}

/// Conversations with no user turn at all, which is `find_last_user_index() == -1` in the
/// reference: every `index > last_user_idx` test passes, so reasoning survives and no
/// `<｜Assistant｜>` is ever emitted. A `usize` port that clamped -1 to 0 would differ
/// here and nowhere else, which is why this case has a test.
#[test]
fn conversations_with_no_user_turn() {
    let thinking = EncodeOpts::new(ThinkingMode::Thinking);
    let only_assistant = json!([{"role": "system", "content": "S"}, {"role": "assistant", "content": "A", "reasoning_content": "R"}]);
    assert_eq!(
        ok(&only_assistant, &thinking),
        "<｜begin▁of▁sentence｜>SR</think>A<｜end▁of▁sentence｜>"
    );

    assert_eq!(
        ok(&json!([{"role": "system", "content": "S"}]), &thinking),
        "<｜begin▁of▁sentence｜>S"
    );

    // An assistant turn with no content at all still emits an empty reasoning block.
    let bare = json!([{"role": "system", "content": "S"}, {"role": "user", "content": "U"}, {"role": "assistant"}]);
    assert_eq!(
        ok(&bare, &thinking),
        "<｜begin▁of▁sentence｜>S<｜User｜>U<｜Assistant｜><think></think><｜end▁of▁sentence｜>"
    );
}

/// The two runtime refusals the reference has, and the boundary's own.
#[test]
fn refusals() {
    let chat = EncodeOpts::new(ThinkingMode::Chat);
    // Fires during ENCODING, not at the boundary: the reference's `assert content` sits
    // inside `render_message`.
    let developer = json!([{"role": "developer", "content": ""}]);
    assert_refused(&enc(&developer, &chat).unwrap_err(), "must have content");
    refuses(&json!({"role": "weird", "content": "x"}), "unknown role");
    refuses(&json!({"content": "x"}), "no `role`");
    // A tool entry must carry a `function` object — the reference indexes it blind.
    refuses(
        &json!({"role": "system", "content": "S", "tools": [{"name": "f"}]}),
        "no `function`",
    );

    // A scalar `content` is refused: defaulting to "" would drop the user's whole
    // question and still leave a prompt that looks perfectly well formed. The ARRAY form
    // is ACCEPTED — see `boundary::openai_array_form_content_is_flattened`.
    refuses(
        &json!({"role": "user", "content": 5}),
        "`content` must be a string",
    );

    refuses(
        &json!({"role": "system", "content": "S", "tools": "f"}),
        "must be an array",
    );
    refuses(
        &json!({"role": "assistant", "content": "A", "tool_calls": "f"}),
        "must be an array",
    );

    // `wo_eos` is tested for TRUTHINESS in the reference (`if wo_eos:`), so `1` counts.
    let truthy = json!([{"role": "user", "content": "P"}, {"role": "assistant", "content": "A", "wo_eos": 1}]);
    assert_eq!(
        ok(&truthy, &chat),
        "<｜begin▁of▁sentence｜><｜User｜>P<｜Assistant｜></think>A"
    );
}

/// The `## Tools` block, in full.
///
/// Hermetic on purpose. `render_tools` is otherwise checked ONLY by gold vectors 1 and 3,
/// which need the checkpoint on disk — so on a machine without it the largest block this
/// module emits, and the only thing that teaches the model the DSML call syntax, would
/// have no coverage at all. The schema line also pins the `json.dumps` spacing
/// (`", "` / `": "`) that `python_json` exists for.
#[test]
fn tools_block_renders_byte_for_byte() {
    // Built through the TYPED api rather than `messages_from_openai`: this is the only
    // test that exercises `Instructions` and `Tool::from_function`, which is how a caller
    // that already holds its tools — the eventual V4 `main` branch — will build them.
    let schema = json!({"name": "f", "description": "D", "parameters": {"type": "object",
        "properties": {"x": {"type": "string"}}, "required": ["x"]}});
    let msgs = vec![
        Message::new(Body::System(Instructions {
            content: "S".into(),
            tools: vec![Tool::from_function(schema)],
            response_format: None,
        })),
        Message::user("U"),
    ];
    assert_eq!(
        encode_messages(msgs, &EncodeOpts::new(ThinkingMode::Thinking)).unwrap(),
        "<｜begin▁of▁sentence｜>S\n\n## Tools\n\nYou have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"<｜DSML｜tool_calls>\" block like the following:\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"$TOOL_NAME\">\n<｜DSML｜parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜parameter>\n...\n</｜DSML｜invoke>\n<｜DSML｜invoke name=\"$TOOL_NAME2\">\n...\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>\n\nString parameters should be specified as is and set `string=\"true\"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\nIf thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.\n\nOtherwise, output directly after </think> with tool calls or final response.\n\n### Available Tool Schemas\n\n{\"name\": \"f\", \"description\": \"D\", \"parameters\": {\"type\": \"object\", \"properties\": {\"x\": {\"type\": \"string\"}}, \"required\": [\"x\"]}}\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n<｜User｜>U<｜Assistant｜><think>"
    );
}

/// Encode a tool call, slice the assistant turn back out, parse it, and get the same
/// `arguments` string back — over the value alphabet that stresses both DSML regexes.
///
/// Hermetic twin of gold vector 1's round-trip. Encode and parse are each pinned against
/// the reference above; only a round trip catches the case where BOTH agree on something
/// the reference does differently.
#[test]
fn encode_then_parse_is_the_identity_on_arguments() {
    for args in [
        r#"{"k": "v"}"#,
        r#"{"k": "a<b"}"#,
        r#"{"k": "say \"hi\""}"#,
        r#"{"k": "back\\slash"}"#,
        r#"{"k": "line1\nline2"}"#,
        r#"{"k": "trailing\n"}"#,
        r#"{"k": "中文 💡"}"#,
        r#"{"n": 5, "b": true, "a": [1, 2], "o": {"z": null}}"#,
        "{}",
    ] {
        let m = json!([
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "reasoning_content": "R", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "f", "arguments": args}}]},
        ]);
        let prompt = ok(&m, &EncodeOpts::new(ThinkingMode::Thinking));
        let marker = "<｜Assistant｜><think>";
        let turn = &prompt[prompt.rfind(marker).unwrap() + marker.len()..];
        let back = parsed(turn, ThinkingMode::Thinking);
        assert_eq!(back.reasoning_content, "R", "{args}");
        assert_eq!(back.content, "", "{args}");
        assert_eq!(back.tool_calls.len(), 1, "{args}");
        assert_eq!(back.tool_calls[0].name, "f", "{args}");
        assert_eq!(back.tool_calls[0].arguments, args, "{args}");
    }
}

/// `arguments` is copied VERBATIM and only `json.loads`-ed at render time, inside a
/// `try` with a bare `except`. So an object, a number, a null or a list all take the
/// wrapper branch and come out as one `arguments` parameter — with `string="false"`,
/// because the original was not a string. Expectations from the reference, 2026-08-06.
#[test]
fn arguments_that_are_not_a_json_string() {
    let chat = EncodeOpts::new(ThinkingMode::Chat);
    let call = |args: Value| {
        json!([
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c", "type": "function", "function": {"name": "f", "arguments": args}}]},
        ])
    };
    for (args, want) in [
        (
            json!({"city": "Paris"}),
            r#"string="false">{"city": "Paris"}<"#,
        ),
        (json!(7), r#"string="false">7<"#),
        (json!(null), r#"string="false">null<"#),
        (json!([1, 2]), r#"string="false">[1, 2]<"#),
        // A STRING that will not parse keeps `string="true"` and its raw text.
        (json!("not json"), r#"string="true">not json<"#),
    ] {
        let out = ok(&call(args.clone()), &chat);
        let want = format!("<{DSML}parameter name=\"arguments\" {want}/{DSML}parameter>");
        assert!(out.contains(&want), "{args}: {out}");
    }
    // Absent entirely is a KeyError in the reference, not a default.
    let missing = json!({"role": "assistant", "content": "", "tool_calls": [
        {"id": "c", "type": "function", "function": {"name": "f"}}]});
    refuses(&missing, "no `function.arguments`");
}
