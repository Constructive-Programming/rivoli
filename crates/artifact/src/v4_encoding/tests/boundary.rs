//! The BOUNDARY's accept/reject edge: every row of [`super::super`]'s over-refusal table, the
//! shapes that are accepted where the reference raises, and the one behaviour that is
//! deliberately not reproduced.
//!
//! See [`super`] for how the expectations were produced and why this module lives in `src/`.

use super::*;
use crate::tokenizer::python_json;

/// OpenAI's array-of-parts `content`, which real clients send constantly.
///
/// **The one shape whose bytes the (unported) differential structurally cannot pin** — a seed
/// carrying them would turn it red, correctly, because on `user` the reference raises and
/// on `system`/`assistant` it emits a Python list repr this deliberately contradicts.
/// `message::text_content` carries the argument; these are the only tests guarding those bytes.
#[test]
fn openai_array_form_content_is_flattened() {
    let chat = EncodeOpts::new(ThinkingMode::Chat);
    let parts = json!([{"type": "text", "text": "hi"}, {"type": "text", "text": "there"}]);
    assert_eq!(
        ok(&json!([{"role": "user", "content": parts}]), &chat),
        "<｜begin▁of▁sentence｜><｜User｜>hi\n\nthere<｜Assistant｜></think>"
    );
    // A single part is the overwhelmingly common form and must be indistinguishable from
    // the plain string — a client that switches spelling must not switch prompts.
    let one = json!([{"role": "user", "content": [{"type": "text", "text": "hi"}]}]);
    assert_eq!(
        ok(&one, &chat),
        ok(&json!([{"role": "user", "content": "hi"}]), &chat)
    );
    // A `tool_result` part keeps its PAYLOAD — this is the arm that separates the outer
    // walker from the inner one, and routing it through the inner one silently deleted a
    // tool's entire output while leaving a well-formed-looking prompt. Byte-verified
    // against the reference's `content_blocks` branch, nested list included, 2026-08-06.
    let tr = json!([{"role": "user", "content": [
        {"type": "text", "text": "q"},
        {"type": "tool_result", "content": "x"},
        {"type": "tool_result", "content": [{"type": "text", "text": "r"}, {"type": "img"}]},
        {"type": "image_url"}]}]);
    assert_eq!(
        ok(&tr, &chat),
        "<｜begin▁of▁sentence｜><｜User｜>q\n\n<tool_result>x</tool_result>\n\n<tool_result>r\n\n[Unsupported img]</tool_result>\n\n[Unsupported image_url]<｜Assistant｜></think>"
    );
    // An image part is NAMED, not dropped: the model should see something was there.
    let img = json!([{"role": "user", "content": [
        {"type": "text", "text": "what is this"},
        {"type": "image_url", "image_url": {"url": "http://x"}}]}]);
    assert!(ok(&img, &chat).contains("what is this\n\n[Unsupported image_url]"));
    // The same on `system`, where the reference emits a Python list repr instead.
    let sys = json!([{"role": "system", "content": [{"type": "text", "text": "S"}]},
                     {"role": "user", "content": "U"}]);
    assert_eq!(
        ok(&sys, &chat),
        "<｜begin▁of▁sentence｜>S<｜User｜>U<｜Assistant｜></think>"
    );
    // An empty array is an empty turn, not an error.
    assert_eq!(
        ok(&json!([{"role": "user", "content": []}]), &chat),
        "<｜begin▁of▁sentence｜><｜User｜><｜Assistant｜></think>"
    );
}

/// Every row of the over-refusal table in [`super::super`]'s header, executed — except the
/// `id` half of the ids row, which [`non_string_fields_are_refused_not_ignored`] owns.
///
/// The table is a claim about this port's accept/reject boundary, and four of its rows
/// were pinned by nothing at all until review counted them — a registry no test reads is
/// a comment that drifts. Each `needle` is the message the code actually produces, so a
/// reworded refusal fails here and gets noticed rather than quietly making the table
/// wrong. Every row is a BOUNDARY refusal, so one loop covers them.
/// The CONTENT half of the header's table: shapes the reference renders through Python's
/// `str()`/`repr()`, where this port refuses rather than guessing which of `True`/`{'a': 1}`/
/// `None` it would have written.
///
/// Two functions rather than one, and split by what the row is ABOUT rather than at an arbitrary
/// line: the single body was 89 lines, which is the shape the code-health gate refuses, and the
/// table is a REGISTRY that reads better as two named values than as the middle of an assertion.
fn content_refusal_rows() -> Vec<(Value, &'static str)> {
    vec![
        // `content` a non-array scalar, and the falsy family the reference maps to "".
        // (An ARRAY is accepted — `openai_array_form_content_is_flattened` — but one
        // whose entries are not objects still fails, inside the block walker.)
        (
            json!({"role": "system", "content": 5}),
            "`content` must be a string",
        ),
        (
            json!({"role": "assistant", "content": [1, 2]}),
            "content block must be an object",
        ),
        (
            json!({"role": "system", "content": true}),
            "`content` must be a string",
        ),
        (
            json!({"role": "assistant", "content": 0}),
            "`content` must be a string",
        ),
        (
            json!({"role": "user", "content": false}),
            "`content` must be a string",
        ),
        // `latest_reminder`, the role with no `or ""` guard anywhere.
        (
            json!({"role": "latest_reminder", "content": 5}),
            "must be a string",
        ),
        // `tool` content that is a scalar OR an object.
        (
            json!({"role": "tool", "content": 5}),
            "must be a string or a block list",
        ),
        (
            json!({"role": "tool", "content": {}}),
            "must be a string or a block list",
        ),
        // A content block's `type`.
        (
            json!({"role": "tool", "content": [{"type": {"a": 1}}]}),
            "`type` must be a string",
        ),
    ]
}

/// The STRUCTURE half: keys whose wrong type would change which tools exist, which expert is
/// called, or the order the results are read in — none of which is text the model merely reads.
fn structure_refusal_rows() -> Vec<(Value, &'static str)> {
    let a_tool = json!([{"type": "function", "function": {"name": "f"}}]);
    vec![
        // `tools`/`tool_calls` present but falsy — on the role that parses the key.
        (
            json!({"role": "system", "content": "S", "tools": {}}),
            "must be an array",
        ),
        (
            json!({"role": "assistant", "content": "A", "tool_calls": 0}),
            "must be an array",
        ),
        // A tool entry whose `function` is not an object — the reference emits the scalar
        // as a schema line. Distinct from `function` ABSENT, which `encode::refusals` covers.
        (
            json!({"role": "system", "content": "S", "tools": [{"function": 5}]}),
            "no `function`",
        ),
        // `function.name` non-string and null take two different paths to a refusal.
        (
            json!({"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": 5, "arguments": "{}"}}]}),
            "`name` must be a string",
        ),
        (
            json!({"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": null, "arguments": "{}"}}]}),
            "`function.name` must be a string",
        ),
        // `reasoning_content`, truthy and falsy.
        (
            json!({"role": "assistant", "content": "A", "reasoning_content": [1]}),
            "`reasoning_content` must be a string",
        ),
        (
            json!({"role": "assistant", "content": "A", "reasoning_content": false}),
            "`reasoning_content` must be a string",
        ),
        // Ids, which are live sort keys in Python.
        (
            json!({"role": "tool", "tool_call_id": 7, "content": "R"}),
            "`tool_call_id` must be a string",
        ),
        // `tools` on a role that renders nothing for them but still flips drop_thinking.
        (
            json!({"role": "assistant", "content": "A", "tools": a_tool}),
            "only `system` and `developer` render a tools block",
        ),
        (
            json!({"role": "latest_reminder", "content": "LR", "tools": a_tool}),
            "only `system` and `developer` render a tools block",
        ),
        // The row the table calls position-dependent: an invalid task is INERT in the
        // reference when `render_message` returns before its assert, and refused here.
        (
            json!({"role": "user", "content": "P", "task": "nope"}),
            "invalid task",
        ),
    ]
}

#[test]
fn the_over_refusal_table_is_real() {
    let mut rows = content_refusal_rows();
    rows.extend(structure_refusal_rows());
    // Anti-vacuity: an empty or truncated table would pass the loop below in silence, and this
    // test's whole subject is that the header's registry has as many live rows as it claims.
    assert_eq!(rows.len(), 20, "the over-refusal table has changed size");
    for (msg, needle) in rows {
        refuses(&msg, needle);
    }
}

/// The measured number-formatting gap from [`super::super`]'s header, pinned so it cannot
/// change silently — and so whoever fixes it has a test to delete.
///
/// The diverging rows are NOT the bytes the reference produces: Python writes `1e-05`,
/// `1e-07`, `1e-30` and the exact 30-digit integer. See the header for why this is
/// recorded rather than approximated.
#[test]
fn numeric_rendering_diverges_from_python() {
    let render = |lit: &str| python_json(&serde_json::from_str::<Value>(lit).unwrap());
    // Agreeing rows — these must never regress.
    for (lit, both) in [
        ("1e30", "1e+30"),
        ("1e16", "1e+16"),
        ("1e-4", "0.0001"),
        ("0.1", "0.1"),
        ("2.0", "2.0"),
        ("9007199254740993", "9007199254740993"),
    ] {
        assert_eq!(
            render(lit),
            both,
            "{lit} should still agree with json.dumps"
        );
    }
    // Diverging rows, with what Python would have written.
    for (lit, ours, python) in [
        ("1e-5", "0.00001", "1e-05"),
        ("1e-7", "1e-7", "1e-07"),
        ("2.5e-8", "2.5e-8", "2.5e-08"),
        ("1e-30", "9.999999999999999e-31", "1e-30"),
        (
            "123456789012345678901234567890",
            "1.2345678901234568e+29",
            "123456789012345678901234567890",
        ),
    ] {
        assert_eq!(render(lit), ours, "{lit}: python writes {python}");
    }

    // The blast radius, so it is visible rather than inferred: a `minimum` of 1e-5 in an
    // ordinary JSON-Schema shifts the bytes of the SYSTEM prompt. The reference writes
    // `"minimum": 1e-05` here (measured 2026-08-06).
    let m = json!([
        {"role": "system", "content": "S", "tools": [{"type": "function", "function":
            {"name": "f", "parameters": {"type": "object",
             "properties": {"x": {"type": "number", "minimum": 1e-5}}}}}]},
        {"role": "user", "content": "U"},
    ]);
    assert!(ok(&m, &EncodeOpts::new(ThinkingMode::Chat)).contains(r#""minimum": 0.00001"#));
}

/// `tools` on a role that cannot render it: refused where the reference would honour it
/// invisibly, ACCEPTED where the reference provably ignores it.
///
/// The split is measured, not reasoned: `merge_tool_messages` rebuilds `user` and `tool`
/// dicts, so the key is gone before `any(m.get("tools") ...)` runs and the encoded output
/// is byte-identical with and without it. `assistant` and `latest_reminder` pass through
/// untouched and do disable `drop_thinking` for the whole conversation. A round-1 version
/// of this check refused all four; review caught it as a disagreement in the REJECT
/// direction, which is the one a strict port is most likely to get wrong.
#[test]
fn tools_on_a_role_that_cannot_render_them() {
    let a_tool = json!([{"type": "function", "function": {"name": "f"}}]);
    for role in ["assistant", "latest_reminder"] {
        let msg = json!({"role": role, "content": "x", "tools": a_tool});
        refuses(&msg, "only `system` and `developer` render a tools block");
    }
    // Inert on these two, so accepted — and the encoded bytes must be unchanged by it.
    let thinking = EncodeOpts::new(ThinkingMode::Thinking);
    for role in ["user", "tool"] {
        let plain = json!([
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U1"},
            {"role": "assistant", "content": "A", "reasoning_content": "R"},
            {"role": role, "content": "x", "tool_call_id": "c"},
        ]);
        let mut with_tools = plain.as_array().unwrap().clone();
        with_tools[3]["tools"] = a_tool.clone();
        assert_eq!(
            ok(&Value::Array(with_tools), &thinking),
            ok(&plain, &thinking),
            "{role}"
        );
    }
    // An empty list is falsy and must not trip it — the same rule as everywhere else.
    assert!(
        message_from_openai(&json!({"role": "assistant", "content": "x", "tools": []})).is_ok()
    );
}

/// Fields read through `as_str()` would turn a wrong TYPE into "absent", and for these
/// three "absent" is a different prompt rather than an error.
#[test]
fn non_string_fields_are_refused_not_ignored() {
    // A mistyped task would quietly stop being a task; the reference asserts on it.
    refuses(
        &json!({"role": "user", "content": "P", "task": 5}),
        "`task` must be a string",
    );
    // …but an explicit null task IS "no task" on both sides.
    assert!(message_from_openai(&json!({"role": "user", "content": "P", "task": null})).is_ok());

    // A non-string id is a live sort key in Python; dropping it to "" would silently
    // reorder the `<tool_result>` blocks.
    let msg = json!({"role": "assistant", "content": "", "tool_calls": [
        {"id": 123, "type": "function", "function": {"name": "f", "arguments": "{}"}}]});
    refuses(&msg, "`id` must be a string");

    // `reasoning_content`. The reference FORMATS a truthy non-string into the block
    // (`<think>[1]</think>`), and the old `as_str()` deleted the block instead.
    let rc = |v: Value| {
        json!([{"role": "user", "content": "U"},
               {"role": "assistant", "content": "A", "reasoning_content": v}])
    };
    let thinking = EncodeOpts::new(ThinkingMode::Thinking);
    let empty_block =
        "<｜begin▁of▁sentence｜><｜User｜>U<｜Assistant｜><think></think>A<｜end▁of▁sentence｜>";
    // Absent and null are `""` on both sides. NOTE these are two DIFFERENT inputs: an
    // earlier version of this test looped over `json!(null)` and `Value::Null`, which
    // are the same value, so it ran the null case twice and covered neither the absent
    // case nor the falsy family below. Caught in review 2026-08-06.
    assert_eq!(
        ok(
            &json!([{"role": "user", "content": "U"}, {"role": "assistant", "content": "A"}]),
            &thinking
        ),
        empty_block
    );
    assert_eq!(ok(&rc(Value::Null), &thinking), empty_block);
    // Every other non-string is refused, truthy or not. The falsy ones are `""` in the
    // reference (`rc = reasoning_content or ""`), so those four are over-refusals —
    // deliberate, uniform with `content`, and carried as their own header table row.
    for v in [
        json!(123),
        json!([1]),
        json!(0),
        json!(false),
        json!([]),
        json!({}),
    ] {
        refuses(
            &rc(v.clone()).as_array().unwrap()[1],
            "`reasoning_content` must be a string",
        );
    }

    // A NULL content on a user turn raises in the reference — the `content or ""` at
    // `render_message` is dead for that role, and the live path joins `None`.
    refuses(
        &json!({"role": "user", "content": null}),
        "null on a `user` message",
    );
    // On an assistant turn it is the ORDINARY OpenAI tool-call shape and must be accepted.
    let tool_call_turn = json!([
        {"role": "user", "content": "U"},
        {"role": "assistant", "content": null, "tool_calls": [
            {"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
    ]);
    assert!(
        ok(&tool_call_turn, &EncodeOpts::new(ThinkingMode::Chat))
            .contains("<｜DSML｜invoke name=\"f\">")
    );
    // And on a system turn it is simply empty.
    assert_eq!(
        ok(
            &json!([{"role": "system", "content": null}]),
            &EncodeOpts::new(ThinkingMode::Chat)
        ),
        "<｜begin▁of▁sentence｜>"
    );
}

/// The failure names WHICH message, not just what was wrong with it.
#[test]
fn boundary_errors_carry_the_message_index() {
    let msgs = json!([{"role": "user", "content": "ok"}, {"role": "user", "content": 5}]);
    let e = messages_from_openai(msgs.as_array().unwrap()).unwrap_err();
    assert!(format!("{e:#}").contains("message 1"), "{e:#}");
}

/// A `tool` message's content: string, block list, or refused.
///
/// The refusals are the interesting half. This role is the only one whose content list
/// the reference genuinely walks, and it has NO `content or ""` guard anywhere on its
/// path — so several shapes that look empty render the word `None`, or raise. Every
/// expectation below was run against the reference on 2026-08-06.
#[test]
fn tool_content_shapes() {
    let chat = EncodeOpts::new(ThinkingMode::Chat);
    let with = |content: Value| json!([{"role": "tool", "tool_call_id": "c", "content": content}]);
    let framed = |body: &str| {
        format!(
            "<｜begin▁of▁sentence｜><｜User｜><tool_result>{body}</tool_result><｜Assistant｜></think>"
        )
    };
    assert_eq!(ok(&with(json!("R")), &chat), framed("R"));
    // Absent is `""` on both sides — `msg.get("content", "")` has a default.
    assert_eq!(
        ok(&json!([{"role": "tool", "tool_call_id": "c"}]), &chat),
        framed("")
    );
    // An EMPTY block list joins to nothing, on both sides.
    assert_eq!(ok(&with(json!([])), &chat), framed(""));
    // A block list joins with `\n\n`; a non-text block is named, not dropped, and both
    // an absent and a null `type` spell the word `None`.
    let blocks = with(json!([
        {"type": "text", "text": "a"}, {"type": "text"},
        {"type": "image"}, {}, {"type": null}
    ]));
    assert_eq!(
        ok(&blocks, &chat),
        framed("a\n\n\n\n[Unsupported image]\n\n[Unsupported None]\n\n[Unsupported None]")
    );

    // NULL, unlike absent, reaches the template as Python's `None` and prints that
    // word — reproduced, per the header's refuse-vs-reproduce rule.
    assert_eq!(ok(&with(json!(null)), &chat), framed("None"));
    // A bare scalar is the same call: `str(True)` is `True`, a word and not JSON.
    for bad in [json!(5), json!(true)] {
        refuses(
            &json!({"role": "tool", "tool_call_id": "c", "content": bad}),
            "must be a string or a block list",
        );
    }
    // Inside a block list: a null/non-string `text` raises TypeError in the reference, a
    // non-object block raises AttributeError, and a non-string `type` interpolates
    // Python's repr (`{'a': 1}`, single quotes).
    for (block, needle) in [
        (
            json!({"type": "text", "text": null}),
            "`text` must be a string",
        ),
        (
            json!({"type": "text", "text": 5}),
            "`text` must be a string",
        ),
        (json!("hello"), "must be an object"),
        (json!({"type": 5}), "`type` must be a string"),
        (json!({"type": {"a": 1}}), "`type` must be a string"),
    ] {
        let msg = json!({"role": "tool", "tool_call_id": "c", "content": [block]});
        refuses(&msg, needle);
    }
}

/// `latest_reminder` is the one role with no `content or ""` on ANY path: absent and
/// null both reach the template as Python's `None` and print that word.
#[test]
fn latest_reminder_content_shapes() {
    let chat = EncodeOpts::new(ThinkingMode::Chat);
    for msg in [
        json!([{"role": "latest_reminder"}]),
        json!([{"role": "latest_reminder", "content": null}]),
    ] {
        assert_eq!(
            ok(&msg, &chat),
            "<｜begin▁of▁sentence｜><｜latest_reminder｜>None"
        );
    }
    // A number interpolates as `5` and a dict as `{'a': 1}` — Python's repr, single
    // quotes and all. Not determined, so refused.
    for bad in [json!(5), json!({"a": 1})] {
        refuses(
            &json!({"role": "latest_reminder", "content": bad}),
            "must be a string",
        );
    }
}
