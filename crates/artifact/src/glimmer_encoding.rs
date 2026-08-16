//! Muse Glimmer's chat framing — a hand-port of the checkpoint's `chat_template.jinja`,
//! pinned byte-for-byte against the template's own renderer.
//!
//! **Why a hand-port at all.** The artifact carries `chat_template.jinja` (this port's
//! converter refuses a checkpoint without it), but nothing in this engine speaks Jinja, and
//! adding an engine for one file is a dependency for what is ultimately string concatenation.
//! Every other model here is hand-ported the same way.
//!
//! Ported from `old:src/artifact/glimmer_encoding.rs` (`wt/glimmer-s2` @ 6b7f496) with the
//! bodies and their comments travelling verbatim — in this repo a comment carries the
//! measurement that justified the choice, so a re-worded one loses evidence. The two helpers
//! it reaches across the crate for, `python_json` and `json_truthy`, live in
//! [`crate::tokenizer`]; the second moved there WITH this port, which is the arrangement that
//! module's header specified in advance.
//!
//! **Why it is pinned against the model's own file rather than against a reading of it.**
//! GLM's hand-port drifted to GLM-4's `<|role|>\n` framing and stayed wrong for months — every
//! benchmark before 2026-08-01 was measured one token off-template per turn — because the only
//! thing it was ever compared against was a second reading by the same author.
//! `crates/artifact/tests/glimmer-chat-cases.json` holds 31 `(kwargs, expected, ids)` triples
//! rendered by `AutoTokenizer.apply_chat_template` on `meta-models/Muse-Glimmer-30B`
//! (`glimmer_template_driver.py` is vendored beside it), and
//! `crates/artifact/tests/glimmer_template.rs` compares this module's bytes against them. A
//! shared misreading cannot survive that.
//!
//! **What framing is FOR, since it looks like decoration.** Glimmer's two stop ids are
//! `<|end_of_text|>` (200001) and `<|eot|>` (200008), and `<|eot|>` is what the template puts
//! at the end of an assistant turn. Fed raw text the model is doing document continuation, is
//! never inside a turn, and has no reason to emit either: that is the exact state behind the
//! old tree's `docs/measurement/benchmarks.md` retraction, where 56 GLM runs ran to their token
//! limit and drifted into looping scaffolding. `<|eom|>` (200007) is NOT a stop id — it ends a
//! message that expects a continuation (the reasoning channel, a non-final tool call), so a
//! decode stops at `<|eot|>` and runs straight through `<|eom|>`.
//!
//! **The output is a STRING, not ids** — unlike [`crate::tokenizer::encode_chat_turns`], which
//! is GLM's and builds a token-ID list. GLM's shape avoids depending on the tokenizer matching
//! specials inside text; here that dependency is carried, and the vendored cases record the
//! ids alongside the string so it can be measured rather than assumed. See
//! `glimmer_template.rs`'s own dated correction for what that pin does and does not establish.
//!
//! > **AMENDED 2026-08-17 (M11b).** That sentence ended "while no Glimmer tokenizer is on this
//! > machine". One is, and the ids are no longer taken on faith:
//! > `rendered_prompts_tokenize_to_the_vendored_ids` runs this module's output through the
//! > shipped tokenizer and compares against `apply_chat_template`'s own ids, 31 of 31.

use serde_json::Value;

use crate::tokenizer::{json_truthy, python_json};

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

impl<'a> GlimmerChatOpts<'a> {
    /// The date, plus the template's own defaults for the other five.
    ///
    /// **This replaced a `Default` impl that set `current_date: ""`, which was the same
    /// contradiction twice over** (review, 2026-08-14): the struct's doc said "no default for
    /// `current_date`" directly above an impl that gave it one, and the value it gave was the
    /// one value that renders a system block **no reference renderer can produce** — the Jinja's
    /// `elif strftime_now is defined` arm always fires under transformers, so `Knowledge cutoff:`
    /// is never followed by a blank line there. `..Default::default()` was the shape the struct's
    /// own doc recommended, so the wrong prompt was the path of least resistance. A constructor
    /// taking the date makes the field unskippable while `..GlimmerChatOpts::new(d)` still costs
    /// a caller nothing.
    ///
    /// Passing `""` deliberately still omits the line; that is a non-template mode, kept because
    /// this is a rendering path with no error channel, and it is now something a caller has to
    /// ask for rather than something they get by writing `..Default::default()`.
    pub fn new(current_date: &'a str) -> Self {
        Self {
            current_date,
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
/// **This closes the SCALAR path only.** A float inside a list, an object or a tool schema goes
/// through [`python_json`] instead — `serde_json`, whose float format is not CPython's — and
/// diverges at `1e-5` (`0.00001` against `1e-05`) and below `1e-6` on the exponent's zero pad
/// (`1e-7` against `1e-07`). That is reachable from an ordinary JSON-Schema `minimum` and it
/// shifts the bytes of the SYSTEM prompt. In the old tree `artifact/dsv4_encoding.rs`'s module
/// header measures the whole table and `numeric_rendering_diverges_from_python` pins it — **and
/// neither has been ported yet**, so on this tree the divergence is recorded here and gated
/// nowhere (PORT NOTE 2026-08-16). It arrives with M8, which owns that file.
///
/// **Not fixed here, on that file's own argument.** Two of its four diverging rows are PARSE
/// precision (`1e-30` → `9.999999999999999e-31`, and an integer past `f64`), which no formatter
/// can reach; the complete fix is `serde_json/arbitrary_precision`, a crate-wide feature that
/// changes `Value::Number` for `format.rs`, the configs and GLM's `tools_system_turn`.
/// Overriding `write_f64` to call this function would close two rows of four and, in that
/// file's words, "look repaired and still be wrong". Recorded here because `py_float` otherwise
/// reads as if the number question were settled; for the scalar path it is (review,
/// 2026-08-14).
///
/// `{:e}` already gives the shortest round-tripping digits in normalized form — mantissa in
/// `[1, 10)` and the decimal exponent spelled out — so this reads them back off it rather than
/// re-deriving them. The hard half of float printing is the digit generation, and Rust's is
/// correct; all that is left is where to put the point.
/// The two values with no normalized form, which is why they are spelled out rather than left
/// to the rule below: a non-finite has no digits and a zero has no leading significant digit.
///
/// `Option` rather than two early returns inside [`py_float`], so that the shared path there is
/// one expression with no bumps in front of it. ONLY zero is degenerate here: non-finite
/// cannot arrive, because the sole route in is `atem_value`'s `Value::Number` arm and
/// `serde_json::Number` cannot hold NaN/Infinity — arms for them would defend an input no
/// caller can construct, which under P7 no gate could ever exercise (review 2026-08-16).
///
/// Zero's SIGN is carried in the bit rather than in the digits — Python prints `-0.0` for
/// negative zero — so `v == 0.0`, which is true for both, is not enough on its own.
fn py_degenerate(v: f64) -> Option<&'static str> {
    match (v == 0.0, v.is_sign_negative()) {
        (true, false) => Some("0.0"),
        (true, true) => Some("-0.0"),
        _ => None,
    }
}

/// Fixed notation: `sig` with the point placed `point` digits after its first digit, padded
/// with zeros on whichever side needs them.
///
/// `point` is the decimal exponent of the leading significant digit, so it is exactly the index
/// of the point relative to that digit — negative means leading zeros, and larger than the digit
/// count means trailing ones.
fn py_fixed(sign: &str, sig: &str, point: i32) -> String {
    let mut s = String::from(sign);
    if point < 0 {
        s.push_str("0.");
        for _ in 0..(-point - 1) {
            s.push('0');
        }
        s.push_str(sig);
        return s;
    }
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
    s
}

/// Scientific notation, with **at least two exponent digits, always signed**: `1e+16`, `1e-05`,
/// `1e+100`. Rust's own `{:e}` writes neither the sign nor the pad, which is one of the four
/// rows in this module's divergence table.
fn py_scientific(sign: &str, sig: &str, point: i32) -> String {
    let mut s = String::from(sign);
    s.push_str(&sig[..1]);
    if sig.len() > 1 {
        s.push('.');
        s.push_str(&sig[1..]);
    }
    s.push_str(&format!(
        "e{}{:02}",
        if point < 0 { '-' } else { '+' },
        point.abs()
    ));
    s
}

fn py_float(v: f64) -> String {
    if let Some(fixed) = py_degenerate(v) {
        return fixed.into();
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
    let sign = if mantissa.starts_with('-') { "-" } else { "" };
    // Normalized and shortest, so there is exactly one digit before the point and no trailing
    // zero to strip: dropping the point leaves the significant digits.
    let sig: String = mantissa
        .trim_start_matches('-')
        .chars()
        .filter(|c| *c != '.')
        .collect();
    // Measured against CPython 3.14, both boundaries: `1e-4` prints `0.0001` and `1e-5` prints
    // `1e-05`; `1e15` prints `1000000000000000.0` and `1e16` prints `1e+16`.
    if (-4..16).contains(&point) {
        py_fixed(sign, &sig, point)
    } else {
        py_scientific(sign, &sig, point)
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
/// id recovery) read it the same way; in the old tree `serve.rs` spelled the `Option` chain out
/// for GLM and jscpd matched the two. That neighbour exists here too —
/// `crates/cli/src/serve/oai.rs::rendered_calls` opens with the same `.get("tool_calls")
/// .and_then(Value::as_array)` — so this factoring is what keeps the duplication gate quiet,
/// not just a readability preference.
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
        // **The reference RAISES here; this port renders `null` instead, deliberately.**
        //
        // > **CORRECTED 2026-08-14, by review, and the correction runs against what this said.**
        // > It read "transformers' sandbox renders it as `null` rather than raising ... `null` is
        // > what the reference produces for it". Measured on the environment that generated the
        // > fixture: `tojson` is `json.dumps(..., ensure_ascii=False)`, and `json.dumps` on a
        // > Jinja `Undefined` raises `TypeError: Object of type Undefined is not JSON
        // > serializable`. So a schema missing `description` or `parameters` fails
        // > `apply_chat_template` outright.
        //
        // Rendering `null` is the same trade `atem` documents for a non-mapping `arguments`: this
        // is a `-> String` path with no error channel, the input is malformed either way, and a
        // caller that reaches here has already decided to send the tools. What is NOT acceptable
        // is claiming the reference agrees, which is the invented-measurement class this repo
        // punishes.
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
    // **A JSON-STRING `arguments` is parsed, not dropped** (review, 2026-08-14). That is what
    // OpenAI, Azure and every SDK mirroring them actually put on the wire, and `as_object` yielded
    // `None` for it and skipped the whole loop — emitting a syntactically valid call to the right
    // function with EVERY argument gone, which then teaches the model on the next turn that
    // calling it without arguments is normal. The template refuses this shape by name (`a JSON
    // string cannot be parsed in the HF jinja sandbox`), and that refusal is about JINJA's
    // sandbox rather than about the shape being meaningless — outside the sandbox it parses fine.
    // `atem`'s doc deferred to "`serve`-side validation"; there was no Glimmer serve path, so
    // the deferral had no destination and the silent drop was the entire behaviour.
    //
    // > **THERE IS ONE NOW, 2026-08-17 (M11b)** — `cli/src/serve/oai.rs::glimmer_prompt`
    // > renders this module for `Arch::MuseGlimmer`, and the parse above stopped being
    // > defence-in-depth the moment that door opened.
    // >
    // > **And it is reachable even though serve withholds `tools`.** That withholding stops the
    // > ATEM *preamble* (the schemas and the worked example), which `GlimmerChatOpts::tools`
    // > gates. It does NOT stop this function: `assistant_calls` runs whenever a MESSAGE in the
    // > history carries `tool_calls`, which is exactly what an OpenAI client replaying a
    // > tool-using conversation sends. So a real request reaches this line with a JSON-string
    // > `arguments`, and without the parse every replayed call would come back to the model as
    // > a correct call with every argument gone.
    // >
    // > The deferral finally has a destination too — refusing a malformed `arguments` belongs
    // > at `serve` — and it is still not written. Parsing the wire format's own spelling is the
    // > behaviour that matters; refusing the rest is a smaller, later question.
    let raw = function_of(call).get("arguments");
    let parsed = raw
        .and_then(Value::as_str)
        .and_then(|s| serde_json::from_str::<Value>(s).ok());
    if let Some(args) = parsed.as_ref().or(raw).and_then(Value::as_object) {
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

/// `tools` as the template sees it: `{%- if tools -%}` is PYTHON truthiness, so an empty array
/// and `null` are both false.
///
/// **`Option::is_some` is the wrong test and it cost 1277 bytes of prompt** (review, 2026-08-14).
/// A caller forwarding an OpenAI request body gets `Some(Array([]))` for `"tools": []` and
/// `Some(Null)` for `"tools": null` — both of which real clients send — and the old gate emitted
/// the whole ATEM preamble for them: an empty `// Tool metadata` list, an empty schema list, and
/// a worked example calling `example_tool_name.example_function_name`, a tool that does not
/// exist. The reference emits none of it. Unreachable from today's binary, which passes `None`;
/// live the moment anything forwards a request body, which is what `atem`'s doc already
/// anticipates.
fn tools_of<'a>(opts: &GlimmerChatOpts<'a>) -> Option<&'a Value> {
    opts.tools.filter(|t| json_truthy(t))
}

/// The tail every system block shares: the optional tool definitions, then the recipients.
fn system_tail(opts: &GlimmerChatOpts) -> String {
    let mut s = String::new();
    if let Some(t) = tools_of(opts) {
        s.push_str("\n\n");
        s.push_str(&tool_defs(t, opts.tool_namespace_descriptions));
    }
    s.push_str("\n\n");
    s.push_str(&system_meta(tools_of(opts)));
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
    // **Last match wins, and the early `return` this replaced was first-match** (review,
    // 2026-08-14). The template's loop carries no `break` and assigns `rns.name` on every hit, so
    // a conversation that reuses a `tool_call_id` attributes the result to the LATER call. A
    // client bug, which is why it is cheap rather than important — but it is wrong in the turn
    // header AND in the `<tool_output name=...>` attribute, so it misattributes twice.
    let mut found: Option<&str> = None;
    for m in all {
        for tc in tool_calls(m) {
            if tc.get("id").and_then(Value::as_str) == Some(id) {
                found = Some(call_name(tc));
            }
        }
    }
    found.unwrap_or(id).into()
}

/// The system block the template SYNTHESISES when the caller supplies no system turn.
///
/// Its own function because it is a different block from the one an explicit `system` message
/// produces — this one carries the assistant persona, the knowledge cutoff and the date, and
/// only the reasoning line and the tail are shared. Collapsing the two would be the change that
/// makes the cutoff line appear on a caller-supplied system turn.
fn synthesised_system(opts: &GlimmerChatOpts) -> String {
    let mut s = String::from("<|start|>system<|message|>You are a helpful AI assistant.");
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
    s
}

/// A caller-supplied `system` turn.
fn system_turn(message: &Value, opts: &GlimmerChatOpts) -> String {
    // Jinja has no case-insensitive replace, so the template spells four realistic casings of
    // "Reasoning effort" and rewrites each to "strength". The suppression below then reads the
    // RESULT, which is why a prompt written with any of the four does not get a second
    // directive appended.
    let mut text = content(message.get("content"));
    for (from, to) in [
        ("Reasoning effort", "Reasoning strength"),
        ("Reasoning Effort", "Reasoning Strength"),
        ("reasoning effort", "reasoning strength"),
        ("REASONING EFFORT", "REASONING STRENGTH"),
    ] {
        text = text.replace(from, to);
    }
    let mut s = String::from("<|start|>system<|message|>");
    s.push_str(&text);
    if !text.to_lowercase().contains("reasoning strength") {
        s.push_str("\n\n");
        s.push_str(&reasoning(opts));
    }
    s.push_str(&system_tail(opts));
    s
}

/// A `tool` result turn. The name appears TWICE — in the turn header and in the
/// `<tool_output>` attribute — which is why [`tool_name`]'s last-match-wins rule misattributes
/// twice when a `tool_call_id` is reused.
fn tool_turn(message: &Value, all: &[Value]) -> String {
    let name = tool_name(message, all);
    let mut s = format!("<|start|>tool {name}<|message|><tool_output name=\"{name}\">\n");
    s.push_str(&content(message.get("content")));
    s.push_str("\n</tool_output><|eot|>");
    s
}

/// An assistant turn that ANSWERS: no tool calls, so one message with a recipient and an end
/// token. Split from [`assistant_calls`] because the two share nothing but the reasoning
/// channel their caller emits before either.
fn assistant_answer(message: &Value) -> String {
    let recipient = message
        .get("recipient")
        .and_then(Value::as_str)
        .filter(|r| !r.is_empty())
        .unwrap_or("user");
    // `end_turn` absent means "infer it": a turn addressed to anyone but the user is a step in
    // a chain, so it ends with `<|eom|>` and the model keeps going. An explicit `false` forces
    // that for a user-addressed turn too, which is how a split answer is fed back.
    // **Only `null`/absent triggers the inference.** The template tests `end_turn is none` and
    // then applies Python truthiness to whatever else is there, so `0`, `""`, `[]` and `{}` END
    // the message and `1`, `"x"`, `"0"`, `"false"` do not. Matching on `Some(Value::Bool)` alone
    // collapsed "present but not a bool" into "absent" and got 10 of 12 shapes wrong (review,
    // 2026-08-14) — and this decides `<|eot|>` against `<|eom|>`, i.e. whether a decode STOPS,
    // since only the first is a stop id.
    let end_turn = match message.get("end_turn") {
        None | Some(Value::Null) => recipient == "user",
        Some(v) => json_truthy(v),
    };
    // The template guards this with `if recipient`, which is always true — the `or 'user'`
    // above cannot yield an empty string — so the guard is dead and is not reproduced.
    let mut s = format!("<|start|>assistant to={recipient}<|message|>");
    s.push_str(&content(message.get("content")));
    s.push_str(if end_turn { "<|eot|>" } else { "<|eom|>" });
    s
}

/// An assistant turn that CALLS: one framed `<atem:function_calls>` block per call.
///
/// `end_token` is the caller's, because it is a fact about the NEXT message rather than about
/// any call — and only the LAST call reads it; every earlier one ends `<|eom|>` because the
/// chain continues.
fn assistant_calls(calls: &[Value], end_token: &str) -> String {
    let mut s = String::new();
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
    s
}

/// A whole `assistant` message: the optional reasoning channel, then either its calls or its
/// answer.
fn assistant_turn(message: &Value, end_token: &str) -> String {
    let mut s = String::new();
    // The reasoning channel is a SEPARATE turn that precedes the answer, and it is emitted
    // whether or not the answer is a tool call.
    if let Some(r) = message.get("reasoning_content").and_then(Value::as_str)
        && !r.is_empty()
    {
        s.push_str(&format!("<|start|>assistant to=self<|message|>{r}<|eom|>"));
    }
    let calls = tool_calls(message);
    if calls.is_empty() {
        s.push_str(&assistant_answer(message));
    } else {
        s.push_str(&assistant_calls(calls, end_token));
    }
    s
}

/// The token that closes a turn: `<|eom|>` when the NEXT message has the same role, else
/// `<|eot|>`.
///
/// Computed per message rather than inside the assistant arm because the template does, and
/// because the condition is about the next message rather than about the turn being framed.
fn end_token(messages: &[Value], i: usize, role: &str) -> &'static str {
    match messages
        .get(i + 1)
        .and_then(|n| n.get("role"))
        .and_then(Value::as_str)
    {
        Some(next) if next == role => "<|eom|>",
        _ => "<|eot|>",
    }
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
        s.push_str(&synthesised_system(opts));
    }
    for (i, message) in messages.iter().enumerate() {
        let role = message.get("role").and_then(Value::as_str).unwrap_or("");
        s.push_str(&match role {
            "system" => system_turn(message, opts),
            "user" => format!(
                "<|start|>user<|message|>{}<|eot|>",
                content(message.get("content"))
            ),
            "tool" => tool_turn(message, messages),
            "assistant" => assistant_turn(message, end_token(messages, i, role)),
            _ => String::new(), // the template has no `else`; see this function's doc
        });
    }
    if opts.add_generation_prompt {
        s.push_str("<|start|>assistant");
    }
    s
}
