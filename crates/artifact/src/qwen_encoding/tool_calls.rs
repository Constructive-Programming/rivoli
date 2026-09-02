//! The `tools` and `tool_calls` halves of Qwen3.8-Flash-Next's chat framing: the system turn's
//! `<tools>` lines and the assistant turn's `<tool_call>` blocks.
//!
//! Split from `qwen_encoding.rs` on 2026-09-02, when that file stood at 799 lines against the 800
//! soft cap; the bodies and their notes moved verbatim, and the module header there still carries
//! the port's argument — including the JSON-string `arguments` divergence [`call_arguments`]
//! implements and `qwen_template.rs` pins against the reference's object form.

use serde_json::Value;

// Qualified through the module, for the reason `qwen_encoding.rs` gives at its own `use`.
use crate::tokenizer;

/// The `tojson` lines a truthy `tools` contributes, or `None` when the template's guard
/// `tools and tools is iterable and tools is not mapping` is false.
///
/// **Python truthiness, not `Option::is_some`.** A caller forwarding an OpenAI request body
/// gets `Some(Array([]))` for `"tools": []` and `Some(Null)` for `"tools": null` — both of
/// which real clients send — and both are FALSE here. Glimmer's port gated on Rust
/// `Option`-ness and emitted 1277 bytes of tool preamble for them (review 2026-08-14); the
/// preamble is larger here.
///
/// An `Object` is truthy but IS a mapping, so it takes the else branch. A non-empty STRING is
/// truthy, iterable and not a mapping, so the TEMPLATE would iterate its CHARACTERS and emit one
/// `"c"` line each, and that is what this arm does. **It is unpinned and unpinnable**:
/// transformers' own `tools` validator raises before the template is entered, so
/// `apply_chat_template` has no opinion to record and no fixture case covers this arm. Kept
/// because the vendored template is the spec this file ports and the arm costs one line; said
/// plainly because "the reference agrees" is the claim this repo punishes.
pub(super) fn tool_json_lines(tools: Option<&Value>) -> Option<Vec<String>> {
    match tools.filter(|t| tokenizer::json_truthy(t))? {
        Value::Array(a) => Some(a.iter().map(tokenizer::python_json).collect()),
        Value::String(s) => Some(
            s.chars()
                .map(|c| tokenizer::python_json(&Value::String(c.to_string())))
                .collect(),
        ),
        _ => None,
    }
}

/// One argument of one call: a JSON string raw, everything else through `tojson`.
///
/// The template's `args_value | string if args_value is string else args_value | tojson | safe`
/// — filters bind tighter than the inline `if`, so a string is emitted unquoted and every other
/// shape is `json.dumps`'d. A port that quoted strings would put `"Oslo"` where the model was
/// trained on `Oslo`.
fn argument_value(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => tokenizer::python_json(other),
    }
}

/// The `arguments` object a call carries, parsing the OpenAI wire format's JSON STRING.
///
/// The reference refuses a string here (`items` on a `str` is a `TypeError`); this port parses
/// it. See the module header for the argument and for the pin that scores the divergence
/// against the reference's own bytes for the object form.
///
/// `arguments is defined and arguments != ''` is the template's guard, so an absent key and the
/// empty string both mean "no parameters" — and `""` parses as nothing anyway, so the guard and
/// the parse agree without a special case.
fn call_arguments(call: &Value) -> Option<Value> {
    let raw = call.get("arguments")?;
    match raw {
        Value::String(s) if s.is_empty() => None,
        Value::String(s) => serde_json::from_str::<Value>(s).ok(),
        other => Some(other.clone()),
    }
}

/// `tool.function` if present, else the object itself — the template's
/// `if tool_call.function is defined: tool_call = tool_call.function`, which accepts both the
/// OpenAI `{"type":"function","function":{…}}` envelope and a bare call object.
fn function_of(call: &Value) -> &Value {
    call.get("function").unwrap_or(call)
}

/// One `<tool_call>` block. `lead` is what separates it from what came before: the first call
/// after non-empty content takes `\n\n`, the first after empty content takes nothing, and every
/// later call takes `\n` — three cases the template spells out and a port is likely to collapse
/// into one.
pub(super) fn tool_call_block(call: &Value, lead: &str) -> String {
    let f = function_of(call);
    let name = f.get("name").and_then(Value::as_str).unwrap_or("");
    let mut s = format!("{lead}<tool_call>\n<function={name}>\n");
    for (k, v) in call_arguments(f)
        .as_ref()
        .and_then(Value::as_object)
        .into_iter()
        .flatten()
    {
        s.push_str(&format!(
            "<parameter={k}>\n{}\n</parameter>\n",
            argument_value(v)
        ));
    }
    s.push_str("</function>\n</tool_call>");
    s
}

/// The calls a message carries — empty unless the template's
/// `tool_calls and is iterable and is not mapping` guard holds on an array.
pub(super) fn tool_calls(message: &Value) -> &[Value] {
    match message.get("tool_calls") {
        Some(Value::Array(a)) => a.as_slice(),
        _ => &[][..],
    }
}
