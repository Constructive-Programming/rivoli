//! Muse Glimmer's chat framing — a hand-port of the checkpoint's `chat_template.jinja`,
//! pinned byte-for-byte against the template's own renderer.
//!
//! **Why a hand-port at all.** The artifact carries `chat_template.jinja` (this port's
//! converter refuses a checkpoint without it), but nothing in this engine speaks Jinja, and
//! adding an engine for one file is a dependency for what is ultimately string concatenation.
//! Every other model here is hand-ported the same way.
//!
//! **Why it is pinned against the model's own file rather than against a reading of it.**
//! GLM's hand-port drifted to GLM-4's `<|role|>\n` framing and stayed wrong for months — every
//! benchmark before 2026-08-01 was measured one token off-template per turn — because the only
//! thing it was ever compared against was a second reading by the same author.
//! `tests/glimmer-chat-cases.json` holds 24 `(kwargs, expected)` triples rendered by
//! `AutoTokenizer.apply_chat_template` on `meta-models/Muse-Glimmer-30B`, and
//! `tests/glimmer_template.rs` compares this module's bytes against them. A shared misreading
//! cannot survive that, which is the same argument `glimmer_reference.rs` makes for the decode
//! loop.
//!
//! **What framing is FOR, since it looks like decoration.** Glimmer's two stop ids are
//! `<|end_of_text|>` (200001) and `<|eot|>` (200008), and `<|eot|>` is what the template puts
//! at the end of an assistant turn. Fed raw text the model is doing document continuation, is
//! never inside a turn, and has no reason to emit either: that is the exact state behind
//! `docs/measurement/benchmarks.md`'s retraction, where 56 GLM runs ran to their token limit
//! and drifted into looping scaffolding. `<|eom|>` (200007) is NOT a stop id — it ends a
//! message that expects a continuation (the reasoning channel, a non-final tool call), so a
//! decode stops at `<|eot|>` and runs straight through `<|eom|>`.
//!
//! **The output is a STRING, not ids.** GLM's port builds from token ids to avoid depending on
//! the tokenizer matching specials inside text; here that dependency is measured rather than
//! avoided — `tests/glimmer_template.rs` pins the ids alongside the string, so a port that
//! emitted a lookalike (`<|start|>` spelled out) would match the text and fail on the ids.

use serde_json::Value;

use super::tokenizer::python_json;

/// The template's own defaults for the system block it synthesises when the caller supplies no
/// system turn. Both are literals in `chat_template.jinja`; neither is in `config.json`.
const DEFAULT_CUTOFF: &str = "2026-01-04";
const DEFAULT_REASONING: &str = "high";

/// `%Y-%m-%d` in UTC for [`GlimmerChatOpts::current_date`], from a caller-supplied instant.
///
/// **The instant is an argument, which is the same argument the opts field makes.** A function
/// that read `SystemTime::now()` internally would be untestable without a clock and would make
/// every caller's output a function of when it ran; taking the time in keeps the decision at the
/// one place that should own it — the CLI, which is where "what day is it" is a legitimate
/// question. `main` passes `SystemTime::now()`; the test passes known epochs.
///
/// Ten lines rather than a dependency: this crate has no date library, `chrono` and `time` are
/// each a nontrivial tree, and what is needed is one civil date with no timezone, no locale and
/// no parsing. The conversion is Howard Hinnant's `civil_from_days`, which is exact for every
/// day in the proleptic Gregorian calendar — no leap-year special cases to get wrong. Before the
/// epoch it returns `1970-01-01`; a system clock set to 1969 is not a case worth branching for.
pub fn utc_date(t: std::time::SystemTime) -> String {
    let secs = t
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    // Shift the era so it starts on 0000-03-01, which puts the leap day at the END of the year
    // and makes every month the same length pattern. 719468 is the day count from that origin to
    // 1970-01-01.
    let z = (secs / 86_400) as i64 + 719_468;
    let era = z / 146_097; // 146097 days = 400 years, the Gregorian cycle
    let doe = z - era * 146_097; // day of era, 0..=146096
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365; // year of era, 0..=399
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // day of the March-started year
    let mp = (5 * doy + 2) / 153; // month, 0 = March
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}")
}

/// What the caller controls, mapping one-to-one onto the template's kwargs.
///
/// **`current_date` is required, and that is a property of the template rather than a house
/// preference.** The Jinja reads `current_date` if defined and otherwise calls
/// `strftime_now('%Y-%m-%d')`, which transformers always provides — so under any real renderer
/// the line is always emitted and its content is the wall clock. A port that read the clock
/// would render differently on either side of midnight, which makes the byte pin a test that
/// fails once a day and makes a served prefix uncacheable across the boundary. Passing the
/// date in moves that decision to the caller, where it can be pinned.
pub struct GlimmerChatOpts<'a> {
    /// The date the system block announces. See the note above on why there is no `Option`.
    pub current_date: &'a str,
    /// Ends the prompt with `<|start|>assistant`, which is what makes the model answer rather
    /// than continue the conversation. Off only for scoring a fixed transcript.
    pub add_generation_prompt: bool,
    /// `None` renders the template's own default, `"high"`. Ignored when the caller supplies a
    /// system turn that already mentions a reasoning strength — see [`render`].
    pub reasoning_strength: Option<&'a str>,
    /// `None` renders [`DEFAULT_CUTOFF`]. Only ever reaches the synthesised system block.
    pub knowledge_cutoff: Option<&'a str>,
    /// OpenAI `tools` — the raw array, each entry either `{"type","function":{...}}` or a bare
    /// function object. This is the ONLY thing that teaches the model the ATEM call syntax;
    /// without it a tool-using prompt gets a model reaching for other frameworks' conventions.
    pub tools: Option<&'a Value>,
    /// `{namespace: description}` for the `// Tool metadata` lines. A namespace absent from
    /// this map gets `""`, which is what the template does with an undefined variable.
    pub tool_namespace_descriptions: Option<&'a Value>,
}

impl Default for GlimmerChatOpts<'_> {
    /// **No default for `current_date`** — `Default` exists for the other five, and a caller
    /// that wants it must still say which day it is. `..Default::default()` on a struct update
    /// leaves the field it does not set, so this costs nothing at a call site.
    fn default() -> Self {
        Self {
            current_date: "",
            add_generation_prompt: true,
            reasoning_strength: None,
            knowledge_cutoff: None,
            tools: None,
            tool_namespace_descriptions: None,
        }
    }
}

/// Python's `str(float)`, which is what Jinja's `{{ v }}` produces for a float and therefore
/// what the model was trained on inside an ATEM parameter.
///
/// **Rust's own formats are all wrong here, each differently**, which is why this is not one
/// call to `format!`. Measured against CPython 3.14:
///
/// | value | Python | `{}` | `{:?}` |
/// |---|---|---|---|
/// | `2.0` | `2.0` | `2` | `2.0` |
/// | `1e15` | `1000000000000000.0` | `1000000000000000` | `1e15` |
/// | `1e16` | `1e+16` | `10000000000000000` | `1e16` |
/// | `1e-5` | `1e-05` | `0.00001` | `1e-5` |
///
/// So `{}` never uses exponent form, `{:?}` uses it far earlier than Python and writes the
/// exponent without a sign or a leading zero, and only Python pads to two exponent digits.
/// The rule CPython applies (`repr` mode, `Py_DTSF_ADD_DOT_0`) is: take the shortest
/// round-tripping digits, then use fixed notation when the decimal exponent is in `-4..16` and
/// scientific otherwise, always with at least one fractional digit in fixed form and at least
/// two exponent digits in scientific form.
///
/// `{:e}` already gives the shortest round-tripping digits in normalized form — mantissa in
/// `[1, 10)` and the decimal exponent spelled out — so this reads them back off it rather than
/// re-deriving them. The hard half of float printing is the digit generation, and Rust's is
/// correct; all that is left is where to put the point.
fn py_float(v: f64) -> String {
    // Non-finite never survives a JSON parse (`serde_json` rejects `NaN`/`Infinity` and the
    // template's `tojson` would too), so the only way to get here is a hand-built `Value`,
    // which `serde_json::Number` also cannot hold. Spelled out rather than left to the
    // formatter, because `{:?}` renders them as `NaN`/`inf` and Python as `nan`/`inf`.
    if !v.is_finite() {
        return if v.is_nan() {
            "nan".into()
        } else if v > 0.0 {
            "inf".into()
        } else {
            "-inf".into()
        };
    }
    // Zero has no normalized form, and its sign is carried in the bit rather than the digits —
    // Python prints `-0.0` for negative zero, so `v == 0.0` (which is true for both) is not
    // enough on its own.
    if v == 0.0 {
        return if v.is_sign_negative() {
            "-0.0".into()
        } else {
            "0.0".into()
        };
    }
    let e = format!("{v:e}"); // "1.2345e3", "-1.5e-7", "5e-324"
    // `LowerExp` on a finite `f64` always writes an `e` and an integer after it. That is not
    // asserted, because this crate denies `expect`/`unwrap` in `src/` and because a panic in a
    // prompt renderer is a worse answer than an unusual one: a formatter that ever stopped
    // doing it would leave the number in its own form rather than take the process down. The
    // exponent is the decimal exponent of the leading significant digit, which is the quantity
    // Python's fixed-vs-scientific rule tests.
    let (mantissa, point) = match e.split_once('e') {
        Some((m, x)) => (m, x.parse::<i32>().unwrap_or(0)),
        None => return e,
    };
    let neg = mantissa.starts_with('-');
    // Normalized and shortest, so there is exactly one digit before the point and no trailing
    // zero to strip: dropping the point leaves the significant digits.
    let sig: String = mantissa
        .trim_start_matches('-')
        .chars()
        .filter(|c| *c != '.')
        .collect();
    let sig = sig.as_str();
    let sign = if neg { "-" } else { "" };
    // Measured against CPython 3.14, both boundaries: `1e-4` prints `0.0001` and `1e-5` prints
    // `1e-05`; `1e15` prints `1000000000000000.0` and `1e16` prints `1e+16`.
    if (-4..16).contains(&point) {
        // Fixed notation. `point` is the index of the decimal point relative to the first
        // significant digit, so it decides how many zeros pad each side.
        let mut s = String::from(sign);
        if point < 0 {
            s.push_str("0.");
            for _ in 0..(-point - 1) {
                s.push('0');
            }
            s.push_str(sig);
        } else {
            let ip = point as usize + 1;
            if sig.len() > ip {
                s.push_str(&sig[..ip]);
                s.push('.');
                s.push_str(&sig[ip..]);
            } else {
                s.push_str(sig);
                for _ in 0..(ip - sig.len()) {
                    s.push('0');
                }
                s.push_str(".0"); // Py_DTSF_ADD_DOT_0: an integral float still shows a fraction
            }
        }
        s
    } else {
        let mut s = String::from(sign);
        s.push_str(&sig[..1]);
        if sig.len() > 1 {
            s.push('.');
            s.push_str(&sig[1..]);
        }
        // At least two exponent digits, always signed: `1e+16`, `1e-05`, `1e+100`.
        s.push_str(&format!(
            "e{}{:02}",
            if point < 0 { '-' } else { '+' },
            point.abs()
        ));
        s
    }
}

/// One value as the ATEM block writes it: a string raw, a bool/null as the bare literal, a
/// list or object through `tojson`, everything else through Jinja's `{{ v }}`.
///
/// The template's own ordering, which matters because a string is also iterable: `boolean`,
/// then `none`, then `mapping or (iterable and not string)`, then the fallthrough. A port that
/// tested "is a container" first would send strings through `tojson` and quote them.
fn atem_value(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Bool(true) => "true".into(),
        Value::Bool(false) => "false".into(),
        Value::Null => "null".into(),
        Value::Array(_) | Value::Object(_) => python_json(v),
        Value::Number(n) => match n.as_f64() {
            // An integral JSON number is an `int` on the Python side and prints without a
            // fraction; only a value that was written as a float takes `py_float`. `is_f64`
            // is what preserves that distinction through the parse.
            Some(f) if n.is_f64() => py_float(f),
            _ => n.to_string(),
        },
    }
}

/// `render_content` — a string as itself, a parts list as its text with image/video parts
/// replaced by their placeholder tokens, `null` (or a missing key) as empty.
fn content(v: Option<&Value>) -> String {
    match v {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .map(|p| match p.get("type").and_then(Value::as_str) {
                Some("image") => "<|patch|>",
                Some("video") => "<|video|>",
                Some("text") => p.get("text").and_then(Value::as_str).unwrap_or(""),
                _ => "",
            })
            .collect(),
        _ => String::new(),
    }
}

/// `tool.function` if present, else the tool object itself — the template's own
/// `fn = tool.function if tool.function is defined else tool`, which accepts both the OpenAI
/// `{"type":"function","function":{...}}` envelope and a bare function object.
fn function_of(tool: &Value) -> &Value {
    tool.get("function").unwrap_or(tool)
}

/// The function name a tool definition or a tool CALL names, or `""` for a malformed one.
///
/// Factored because the same chain is read from three places — the ATEM block, the `to=`
/// recipient, and the `tool_call_id` recovery — and jscpd reported two of them as clones the
/// first time this file compiled. Rendering a nameless call as `to=` is the template's own
/// behaviour: `tc.function.name` on a missing key yields the empty string in the sandbox.
fn call_name(call: &Value) -> &str {
    function_of(call)
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("")
}

/// The tool calls a message carries — empty when the key is absent, `null`, or not an array.
///
/// A slice rather than an `Option<&Vec<_>>` so the two callers (`render`'s branch test and the
/// id recovery) read it the same way; `serve.rs` spells the `Option` chain out for GLM and
/// jscpd matched the two.
fn tool_calls(message: &Value) -> &[Value] {
    message
        .get("tool_calls")
        .and_then(Value::as_array)
        .map_or(&[][..], Vec::as_slice)
}

/// Every tool namespace in first-seen order: `fn.name.split('.')[0]`.
///
/// **Not "the part before a dot, if there is one".** The template splits unconditionally and
/// takes element 0, so a function with no dot is its own namespace — `get_weather` yields the
/// recipient `"get_weather.*"`, which reads odd and is what the model was trained on.
fn namespaces(tools: &Value) -> Vec<&str> {
    let mut seen: Vec<&str> = Vec::new();
    for tool in tools.as_array().into_iter().flatten() {
        let Some(name) = function_of(tool).get("name").and_then(Value::as_str) else {
            continue;
        };
        let ns = name.split('.').next().unwrap_or(name);
        if !seen.contains(&ns) {
            seen.push(ns);
        }
    }
    seen
}

/// `render_tool_defs` — the ATEM protocol preamble, the namespace metadata, the JSON schemas,
/// and the worked example. Every literal here is `chat_template.jinja`'s, byte for byte.
fn tool_defs(tools: &Value, descriptions: Option<&Value>) -> String {
    let mut s = String::from(
        "In this environment you have access to a set of tools you can use to answer the \
         user's question.\n\nYou can invoke a function by writing a \"<atem:function_calls>\" \
         block like the following:\n<atem:function_calls>\n<atem:invoke name=\"$FUNCTION_NAME\">\
         \n<atem:parameter name=\"$PARAMETER_NAME\">$PARAMETER_VALUE</atem:parameter>\n...\n\
         </atem:invoke>\n</atem:function_calls>\n\nString and scalar parameters should be \
         specified as is, while lists and objects should use JSON format. Note that spaces for \
         string values are not stripped. The output is not expected to be valid XML and is \
         parsed with regular expressions.\nHere are the functions available in JSONSchema \
         format:\n// Tool metadata\n",
    );
    for ns in namespaces(tools) {
        let desc = descriptions
            .and_then(|d| d.get(ns))
            .cloned()
            .unwrap_or(Value::String(String::new()));
        s.push_str(&format!(
            "{{\"name\": {}, \"description\": {}}}\n",
            python_json(&Value::String(ns.into())),
            python_json(&desc)
        ));
    }
    s.push_str("// Function schemas");
    for tool in tools.as_array().into_iter().flatten() {
        let f = function_of(tool);
        // `| tojson` on an undefined variable is a Jinja error, but transformers' sandbox
        // renders it as `null` rather than raising; a schema missing `description` or
        // `parameters` is malformed input either way, and `null` is what the reference
        // produces for it.
        s.push_str(&format!(
            "\n{{\"name\": {}, \"description\": {}, \"parameters\": {}}}",
            python_json(f.get("name").unwrap_or(&Value::Null)),
            python_json(f.get("description").unwrap_or(&Value::Null)),
            python_json(f.get("parameters").unwrap_or(&Value::Null))
        ));
    }
    s.push_str(
        "\n\nHere's an example of how to call a function in the tool set:\n(If the tool \
         namespace is not specified, invoke the function directly as `example_function_name` \
         rather than `example_tool_name.example_function_name`)\n\n\
         to=example_tool_name.example_function_name\n\n<atem:function_calls>\n\
         <atem:invoke name=\"example_tool_name.example_function_name\">\n\
         <atem:parameter name=\"example_parameter_1\">value_1</atem:parameter>\n\
         <atem:parameter name=\"example_parameter_2\">This is the value for the second \
         parameter\nthat can span\n\"multiple\" lines\n</atem:parameter>\n</atem:invoke>\n\
         </atem:function_calls>",
    );
    s
}

/// `render_system_meta` — the recipient whitelist that closes every system block.
fn system_meta(tools: Option<&Value>) -> String {
    let mut r = vec![String::from("\"self\"")];
    if let Some(t) = tools {
        r.extend(namespaces(t).into_iter().map(|ns| format!("\"{ns}.*\"")));
    }
    r.push(String::from("\"user\""));
    format!("# Valid recipients: {}.", r.join(", "))
}

/// `render_atem` — one tool call as the `<atem:function_calls>` block.
///
/// **Arguments must be an object.** The template raises on anything else, with a message
/// naming the reason: a JSON *string* of arguments — which is what the OpenAI wire format
/// actually carries — cannot be parsed inside the Jinja sandbox. Here that is a silent empty
/// block instead of a panic, because this is a rendering path and the caller has already
/// decided to send the call; `serve`-side validation is where a refusal belongs.
fn atem(call: &Value) -> String {
    let mut s = format!(
        "<atem:function_calls>\n<atem:invoke name=\"{}\">\n",
        call_name(call)
    );
    if let Some(args) = function_of(call)
        .get("arguments")
        .and_then(Value::as_object)
    {
        for (k, v) in args {
            s.push_str(&format!(
                "<atem:parameter name=\"{k}\">{}</atem:parameter>\n",
                atem_value(v)
            ));
        }
    }
    s.push_str("</atem:invoke>\n</atem:function_calls>");
    s
}

/// The system block's reasoning line, shared by the synthesised block and the explicit one.
fn reasoning(opts: &GlimmerChatOpts) -> String {
    let rs = opts
        .reasoning_strength
        .filter(|s| !s.is_empty())
        .unwrap_or(DEFAULT_REASONING);
    format!("Reasoning strength: {rs}.")
}

/// The tail every system block shares: the optional tool definitions, then the recipients.
fn system_tail(opts: &GlimmerChatOpts) -> String {
    let mut s = String::new();
    if let Some(t) = opts.tools {
        s.push_str("\n\n");
        s.push_str(&tool_defs(t, opts.tool_namespace_descriptions));
    }
    s.push_str("\n\n");
    s.push_str(&system_meta(opts.tools));
    s.push_str("<|eot|>");
    s
}

/// The name a `tool` message reports, recovering it from the conversation when the message
/// carries only a `tool_call_id`.
///
/// The template's fallback chain, exactly: an explicit `name`; else the id resolved against
/// every earlier tool call; else the raw id; else empty. The last two arms are why a stray id
/// renders as `<|start|>tool nosuch` rather than failing — pinned by the
/// `tool_result_unresolved` case.
fn tool_name(message: &Value, all: &[Value]) -> String {
    if let Some(n) = message.get("name").and_then(Value::as_str)
        && !n.is_empty()
    {
        return n.into();
    }
    let Some(id) = message.get("tool_call_id").and_then(Value::as_str) else {
        return String::new();
    };
    for m in all {
        for tc in tool_calls(m) {
            if tc.get("id").and_then(Value::as_str) == Some(id) {
                return call_name(tc).into();
            }
        }
    }
    id.into()
}

/// Render an OpenAI-shaped `messages` array as Muse Glimmer's chat framing.
///
/// **Messages are `Value`, not a typed model.** The template branches on seven optional fields
/// (`reasoning_content`, `tool_calls`, `recipient`, `end_turn`, `name`, `tool_call_id`, and
/// content-as-parts), most of which no caller sets; a struct carrying all of them would be
/// more code than reading the keys, and the pinned cases are themselves JSON.
///
/// **An unrecognised role renders NOTHING.** The template's `if/elif` chain has no `else`, so
/// a `developer` turn silently disappears. That is a deliberate divergence from GLM's port,
/// which frames an unknown role as `user` on the argument that dropping content is worse —
/// here the model's own file makes the other choice, and matching it is the whole point. The
/// `unknown_role` case pins it.
pub fn render(messages: &[Value], opts: &GlimmerChatOpts) -> String {
    let mut s = String::from("<|begin_of_text|>");
    let has_system = messages
        .iter()
        .any(|m| m.get("role").and_then(Value::as_str) == Some("system"));
    if !has_system {
        s.push_str("<|start|>system<|message|>You are a helpful AI assistant.");
        let kc = opts
            .knowledge_cutoff
            .filter(|c| !c.is_empty())
            .unwrap_or(DEFAULT_CUTOFF);
        s.push_str(&format!("\nKnowledge cutoff: {kc}."));
        if !opts.current_date.is_empty() {
            s.push_str(&format!("\nCurrent date: {}.", opts.current_date));
        }
        s.push_str("\n\n");
        s.push_str(&reasoning(opts));
        s.push_str(&system_tail(opts));
    }
    for (i, message) in messages.iter().enumerate() {
        let role = message.get("role").and_then(Value::as_str).unwrap_or("");
        // Only the tool-call arm reads this, and only its LAST call does. It is computed here
        // rather than there because the template does, and because the condition is about the
        // NEXT message rather than about the call.
        let end_token = match messages
            .get(i + 1)
            .and_then(|n| n.get("role"))
            .and_then(Value::as_str)
        {
            Some(next) if next == role => "<|eom|>",
            _ => "<|eot|>",
        };
        match role {
            "system" => {
                // Jinja has no case-insensitive replace, so the template spells four realistic
                // casings of "Reasoning effort" and rewrites each to "strength". The
                // suppression below then reads the RESULT, which is why a prompt written with
                // any of the four does not get a second directive appended.
                let mut text = content(message.get("content"));
                for (from, to) in [
                    ("Reasoning effort", "Reasoning strength"),
                    ("Reasoning Effort", "Reasoning Strength"),
                    ("reasoning effort", "reasoning strength"),
                    ("REASONING EFFORT", "REASONING STRENGTH"),
                ] {
                    text = text.replace(from, to);
                }
                s.push_str("<|start|>system<|message|>");
                s.push_str(&text);
                if !text.to_lowercase().contains("reasoning strength") {
                    s.push_str("\n\n");
                    s.push_str(&reasoning(opts));
                }
                s.push_str(&system_tail(opts));
            }
            "user" => {
                s.push_str("<|start|>user<|message|>");
                s.push_str(&content(message.get("content")));
                s.push_str("<|eot|>");
            }
            "tool" => {
                let name = tool_name(message, messages);
                s.push_str(&format!(
                    "<|start|>tool {name}<|message|><tool_output name=\"{name}\">\n"
                ));
                s.push_str(&content(message.get("content")));
                s.push_str("\n</tool_output><|eot|>");
            }
            "assistant" => {
                // The reasoning channel is a SEPARATE turn that precedes the answer, and it is
                // emitted whether or not the answer is a tool call.
                if let Some(r) = message.get("reasoning_content").and_then(Value::as_str)
                    && !r.is_empty()
                {
                    s.push_str(&format!("<|start|>assistant to=self<|message|>{r}<|eom|>"));
                }
                let calls = tool_calls(message);
                if !calls.is_empty() {
                    for (j, tc) in calls.iter().enumerate() {
                        s.push_str(&format!(
                            "<|start|>assistant to={}<|message|>",
                            call_name(tc)
                        ));
                        s.push_str(&atem(tc));
                        s.push_str(if j + 1 == calls.len() {
                            end_token
                        } else {
                            "<|eom|>"
                        });
                    }
                } else {
                    let recipient = message
                        .get("recipient")
                        .and_then(Value::as_str)
                        .filter(|r| !r.is_empty())
                        .unwrap_or("user");
                    // `end_turn` absent means "infer it": a turn addressed to anyone but the
                    // user is a step in a chain, so it ends with `<|eom|>` and the model keeps
                    // going. An explicit `false` forces that for a user-addressed turn too,
                    // which is how a split answer is fed back.
                    let end_turn = match message.get("end_turn") {
                        Some(Value::Bool(b)) => *b,
                        _ => recipient == "user",
                    };
                    // The template guards this with `if recipient`, which is always true —
                    // the `or 'user'` above cannot yield an empty string — so the guard is
                    // dead and is not reproduced.
                    s.push_str(&format!("<|start|>assistant to={recipient}<|message|>"));
                    s.push_str(&content(message.get("content")));
                    s.push_str(if end_turn { "<|eot|>" } else { "<|eom|>" });
                }
            }
            _ => {} // the template has no `else`; see this function's doc
        }
    }
    if opts.add_generation_prompt {
        s.push_str("<|start|>assistant");
    }
    s
}
