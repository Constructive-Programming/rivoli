//! Qwen3.8-Flash-Next's chat framing — a hand-port of the checkpoint's own
//! `chat_template.jinja`, pinned byte-for-byte and id-for-id against the template's own
//! renderer.
//!
//! **Why a hand-port at all.** Nothing in this engine speaks Jinja, and adding an engine for
//! one file is a dependency for what is ultimately string concatenation. Every other model
//! here is hand-ported the same way ([`crate::glimmer_encoding`], [`crate::v4_encoding`],
//! [`crate::tokenizer`]'s GLM surface).
//!
//! **Why it is pinned against the model's own file rather than against a reading of it.**
//! GLM's hand-port drifted to GLM-4's `<|role|>\n` framing and stayed wrong for months —
//! every benchmark before 2026-08-01 was measured one token off-template per turn — because
//! the only thing it was ever compared against was a second reading by the same author.
//! `crates/artifact/tests/qwen-chat-cases.json` holds `(kwargs, expected, ids)` triples
//! rendered by `AutoTokenizer.apply_chat_template` at the pinned revision
//! (`qwen_template_driver.py` is vendored beside it), and
//! `crates/artifact/tests/qwen_template.rs` compares this module's bytes AND the ids they
//! tokenize to against them. A shared misreading cannot survive that.
//!
//! **The template is vendored in tree** at `docs/measurement/qwen-reference/chat_template.jinja`
//! — 8,952 bytes, sha256 `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`,
//! fnv1a64 `8b6b0871c5db260b`, fetched at revision `236dfdf285828023ca3bcd3f37366c58a3469b13`.
//! It ships in the FP8 repo in **two** places and they are **exactly byte-identical**, not
//! identical-modulo-a-newline: `chat_template.jinja` (8,952 B, no trailing newline) and
//! `tokenizer_config.json`'s `chat_template` key (8,952 chars). Measured 2026-08-31 and gated
//! by `qwen_template.rs::the_vendored_template_is_the_pinned_revisions_own_file`, which
//! recomputes both hashes from the file on disk — a pin checked only against a frozen copy of
//! itself is decoration.
//!
//! **What framing is FOR, since it looks like decoration.** The stop ids are `<|im_end|>`
//! (248046) and `<|endoftext|>` (248044) — `generation_config.json`'s `eos_token_id` is
//! `[248046, 248044]`. `<|im_end|>` is what closes a turn. Fed raw text the model is doing
//! document continuation, is never inside a turn, and has no reason to emit either: that is
//! the state behind the old tree's `docs/measurement/benchmarks.md` retraction, where 56 GLM
//! runs ran to their token limit and drifted into looping scaffolding.
//!
//! **The output is a STRING, not ids** — like [`crate::glimmer_encoding`] and unlike
//! [`crate::tokenizer::encode_chat_turns`], which is GLM's. So this module depends on the
//! tokenizer matching `<|im_start|>`, `<think>` and friends as single ids, and that dependency
//! is measured rather than assumed: `<|im_start|>`/`<|im_end|>` are `special: true` added
//! tokens and `<think>`/`</think>`/`<tool_call>`/`<tool_response>` are `special: false` ones
//! (`tokenizer_config.json`'s `added_tokens_decoder`, all `normalized: false`), and the id pin
//! runs this module's bytes through the shipped 12,809,320 B `tokenizer.json`.
//!
//! **There is no BOS.** `tokenizer_config.json` carries `bos_token: null` and
//! `add_bos_token: false`, and the template emits no document-open token — a rendered prompt
//! starts with `<|im_start|>`. A port that prepended one (four of the five architectures here
//! have one) would shift every id.
//!
//! # What this port does NOT reproduce
//!
//! Recorded here rather than left to a reader, because "the reference agrees" is the
//! invented-measurement class this repo punishes:
//!
//! - **A `tool_call.arguments` that is a JSON STRING** — which is what OpenAI, Azure and every
//!   SDK mirroring them actually put on the wire — makes the reference raise: `arguments|items`
//!   is Jinja's `items` filter, which is `TypeError: Can only get item pairs from a mapping.`
//!   on a `str`. This port PARSES it and renders the resulting object, and the divergence is
//!   pinned against the reference rather than asserted: `qwen_template.rs::
//!   a_json_string_arguments_renders_as_the_object_the_reference_was_given` renders the string
//!   form and compares it to the *object* form's vendored reference bytes. The alternative was
//!   refusing every replayed tool conversation, and the Glimmer port measured the third option
//!   (drop the arguments silently) as teaching the model that calling a tool with no arguments
//!   is normal.
//! - **Python's `repr` for floats inside `tojson`.** [`crate::tokenizer::python_json`] is `serde_json` with
//!   Python's separators, and its number path diverges from `json.dumps` on small-magnitude
//!   exponent forms (`1e-5` → `0.00001` against `1e-05`) and on parse-precision extremes. That
//!   table is already measured and gated by
//!   `v4_encoding::tests::boundary::numeric_rendering_diverges_from_python`; it is cited rather
//!   than re-derived, and this module's fixture pins only the agreeing rows (`2.0`, `0.1`,
//!   `1e16`, `1e30`, `1e-4`, integers). The complete fix is `serde_json/arbitrary_precision`,
//!   crate-wide, which would re-run GLM's gold ids — recorded there, owned there.
//! - **Three shapes the reference cannot be ASKED about, so nothing here is scored against it.**
//!   `apply_chat_template` refuses `messages=[]` (`ValueError: Cannot apply chat template to an
//!   empty conversation.`) and a MAPPING `tools` (`ValueError: Tools should either be a JSON
//!   schema, …`) before the template is entered, and a non-empty STRING `tools` hits the same
//!   tools validator — all three measured 2026-08-31 against the pinned revision. What this port
//!   does is what the TEMPLATE's own text says: `No messages provided.` for the first, the else
//!   branch for a mapping (`is not mapping` fails), and for a string the tools branch iterating
//!   its CHARACTERS, since the guard is Python truthiness + iterable + not-mapping. That last
//!   one is the template's behaviour reproduced WITHOUT a reference pin, which is the honest
//!   description of it; no fixture case covers it and none can.
//! - **A `tool_calls` value that is a non-empty STRING.** It reaches `tool_call.name` on a `str`
//!   and raises Jinja's own `UndefinedError` — an internal error rather than a template-authored
//!   refusal, so it is not a contract and this port treats it as absent.
//! - **The exact refusal text for a non-string `reasoning_effort`.** The reference renders the
//!   offending value with Jinja's `~`, which is Python `str()` — `None`, `True`, `[1, 2]`. This
//!   port reproduces that for null, booleans and numbers and falls back to JSON for containers.
//!   Every one of those shapes is a refusal either way; only the message text is approximate.

use anyhow::{Result, bail};
use serde_json::Value;

// Qualified through the module rather than imported by name. Two reasons, and the second is a
// gate: `python_json` and `json_truthy` are Python semantics owned by no model (see
// `tokenizer.rs`'s own headers), so naming the module at every call site says where they come
// from — and `v4_encoding/render.rs`'s import run has the identical SHAPE, which jscpd matched
// at 29 tokens the first time this file compiled. That file's comment predicts the clone and
// asks the next port not to re-create it; a braceless `use` is how this one does not.
use crate::tokenizer;

mod tool_calls;
use tool_calls::{tool_call_block, tool_calls, tool_json_lines};

/// `<|im_start|>`, id 248045.
pub const IM_START: &str = "<|im_start|>";
/// `<|im_end|>`, id 248046 — `generation_config.json`'s first `eos_token_id`.
pub const IM_END: &str = "<|im_end|>";
/// `<think>`, id 248068. An added token with `special: false`, so `skip_special_tokens` does
/// NOT drop it from a decode — which is what makes the reply readable back with
/// `serve::oai::split_think`.
pub const THINK_OPEN: &str = "<think>";
/// `</think>`, id 248069.
pub const THINK_CLOSE: &str = "</think>";

/// The template's own reasoning-effort instruction for `xhigh`, its DEFAULT.
///
/// **This is emitted on a plain `[{"role":"user"}]` render with no kwargs at all**, because
/// `reasoning_effort|default('xhigh')` resolves to `xhigh` and thinking is the default. So the
/// bare-minimum prompt carries a synthesised system turn; a port that treated "no kwargs" as
/// "no system block" would be short one turn on every single request.
const EFFORT_XHIGH: &str = "Reasoning effort is set to xhigh. Please think carefully through \
     the task, validate key assumptions, consider plausible alternatives, and prioritize \
     correctness, consistency, and clarity in the final answer.";
/// The template's instruction for `low`.
const EFFORT_LOW: &str = "Reasoning effort is set to low. Keep your thinking brief and focused, \
     moving directly to the conclusion without unnecessary elaboration.";

/// The tool-protocol preamble, verbatim from `chat_template.jinja`. One literal because the
/// template emits it as one literal.
const TOOLS_PREAMBLE: &str = "# Tools\n\nYou have access to the following functions:\n\n<tools>";

/// The call-format instructions that close the tools system turn, verbatim.
const TOOLS_FORMAT: &str = "\n\nIf you choose to call a function ONLY reply in the following \
     format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n\
     <parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\n\
     This is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n\
     </function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the \
     specified format: an inner <function=...></function> block must be nested within \
     <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may \
     provide optional reasoning for your function call in natural language BEFORE the function \
     call, but NOT after\n- If there is no function call available, answer the question like \
     normal with your current knowledge and do not tell the user about function calls\n\
     </IMPORTANT>";

/// A Jinja kwarg as the template's three tests see it: `is undefined`, `is true`, `is false`.
///
/// **Four states, not two, and the fourth is the trap.** Jinja's `true`/`false` tests are
/// `value is True` / `value is False` — IDENTITY against the two singletons, not truthiness —
/// so `enable_thinking = 1`, `= 0`, `= null` and `= "true"` are none of undefined, true or
/// false. The template asks `enable_thinking is undefined or enable_thinking is true` for the
/// reasoning instructions and `enable_thinking is defined and enable_thinking is false` for the
/// generation prompt's think block, so those values land in a state neither branch was written
/// for: **no reasoning instructions, but an OPEN `<think>`**. Collapsing this to a `bool` gets
/// that asymmetric state wrong in the SYSTEM prompt, which is the same class of error as
/// Glimmer's `end_turn` (10 of 12 shapes wrong, review 2026-08-14).
///
/// Shared by `enable_thinking` and `preserve_thinking` because the template asks both of them
/// the same `is undefined or is true` question.
// `Ord` so a test can collect the four states into a `BTreeSet` and require every one of them
// by name; the ordering itself carries no meaning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum JinjaFlag {
    /// The kwarg was not passed. `is undefined` is true.
    Undefined,
    /// The kwarg is the boolean `true`. `is true` is true.
    True,
    /// The kwarg is the boolean `false`. `is false` is true.
    False,
    /// The kwarg is defined and is neither boolean — `1`, `0`, `null`, `"true"`, `[]`. All
    /// three tests are false, which is what makes this its own state.
    OtherDefined,
}

impl JinjaFlag {
    /// The state a JSON kwarg lands in. `None` is an absent key, which is `undefined`.
    pub fn of(kwarg: Option<&Value>) -> Self {
        match kwarg {
            None => Self::Undefined,
            Some(Value::Bool(true)) => Self::True,
            Some(Value::Bool(false)) => Self::False,
            Some(_) => Self::OtherDefined,
        }
    }

    /// The template's `x is undefined or x is true`.
    fn undefined_or_true(self) -> bool {
        matches!(self, Self::Undefined | Self::True)
    }

    /// The template's `x is defined and x is false`.
    fn defined_and_false(self) -> bool {
        matches!(self, Self::False)
    }
}

/// What the caller controls, mapping one-to-one onto the template's kwargs.
///
/// **No `current_date` and no clock.** Unlike Glimmer's template this one calls no
/// `strftime_now`, so a render is a pure function of its inputs and the byte pin cannot go red
/// at midnight.
pub struct QwenChatOpts<'a> {
    /// Ends the prompt with `<|im_start|>assistant\n` plus a think block, which is what makes
    /// the model answer rather than continue the conversation. Python truthiness in the
    /// template; a `bool` here because no other shape changes the outcome.
    pub add_generation_prompt: bool,
    /// Thinking is the DEFAULT: [`JinjaFlag::Undefined`] renders the reasoning instructions and
    /// leaves `<think>` open. Only [`JinjaFlag::False`] closes it as an empty block.
    pub enable_thinking: JinjaFlag,
    /// Whether prior assistant turns replay their `reasoning_content`.
    /// [`JinjaFlag::Undefined`] and [`JinjaFlag::True`] replay every turn; anything else
    /// replays only turns AFTER the last real user query.
    pub preserve_thinking: JinjaFlag,
    /// `xhigh` (the default), `medium` or `low`; anything else is refused with the template's
    /// own message. `None` is the absent kwarg, which resolves to `xhigh` — **not** to "no
    /// instruction". A JSON `null` is a `Some(Value::Null)` here and is REFUSED, because
    /// Jinja's `default` filter replaces only `undefined`.
    pub reasoning_effort: Option<&'a Value>,
    /// OpenAI `tools` — the raw array, each entry rendered through `tojson` as-is (this
    /// template does NOT unwrap the `{"type","function":{…}}` envelope, unlike GLM's).
    pub tools: Option<&'a Value>,
    /// Prefixes each vision placeholder with `Picture N: ` / `Video N: `. The template's own
    /// kwarg; off by default.
    pub add_vision_id: bool,
}

impl Default for QwenChatOpts<'_> {
    /// The template's own defaults — every field the value a render with no kwargs uses.
    ///
    /// `add_generation_prompt` is `false` because that is `apply_chat_template`'s default and
    /// therefore what the vendored cases render when they omit it. A serving caller sets it.
    fn default() -> Self {
        Self {
            add_generation_prompt: false,
            enable_thinking: JinjaFlag::Undefined,
            preserve_thinking: JinjaFlag::Undefined,
            reasoning_effort: None,
            tools: None,
            add_vision_id: false,
        }
    }
}

/// Jinja's `trim` filter, which is Python's `str.strip()` — **not** Rust's `str::trim()`.
///
/// The two differ on exactly four codepoints: U+001C–U+001F (the ASCII file/group/record/unit
/// separators) satisfy Python's `str.isspace()` because their bidirectional class is `B`/`S`,
/// and are NOT in Unicode's `White_Space` property, so Rust leaves them. Everything else
/// agrees, NBSP and the ideographic space included. Measured against CPython 3.14.6 by
/// `qwen_template_driver.py`'s `whitespace_trim` case, which puts all four in a user turn so
/// the reference's own answer is what this is scored against.
///
/// This matters at every content site: the template applies `|trim` to **all four**
/// `render_content` call sites — the system turn, the reverse user-query scan, and every
/// message in the main loop — plus to `reasoning_content`. A port that skipped it would
/// diverge on any content with a trailing newline, which is most content pasted by a human.
fn jinja_trim(s: &str) -> &str {
    s.trim_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c))
}

/// The render-wide vision state: the two counters the template keeps in `namespace()`s, plus
/// the `add_vision_id` kwarg that decides whether they are ever printed.
///
/// One struct rather than three parameters because they travel together through every
/// `render_content` call and the counters are genuinely shared mutable state — the template's
/// `image_count` is one namespace for the whole render, not one per message.
struct Vision {
    image: u32,
    video: u32,
    add_id: bool,
}

/// One `content` part, as `render_content`'s inner `for` branches on it.
///
/// The template's own ordering, which is load-bearing because a part can match more than one
/// arm: `'image' in item or 'image_url' in item or item.type == 'image'`, then the video pair,
/// then `'text' in item`, then raise. `in` on a dict is KEY membership, so `{"image_url": …}`
/// is an image by its key alone and never reaches the text arm.
enum Part {
    Media(Media),
    Text,
    Unexpected,
}

impl Part {
    fn of(item: &Value) -> Self {
        if media_kind(item, "image", &["image", "image_url"]) {
            Self::Media(Media::Image)
        } else if media_kind(item, "video", &["video"]) {
            Self::Media(Media::Video)
        } else if item.get("text").is_some() {
            Self::Text
        } else {
            Self::Unexpected
        }
    }
}

/// Does this item carry the media kind `tag`?
///
/// The template asks one question in two spellings: a part names its kind in `type`, or it just
/// carries the payload key. Spelled as `||` chains the answer is three-operand conditionals the
/// reviewer counts as branches; spelled as this predicate, each kind is a tag and a key list, and
/// the ORDER of the arms in [`Part::of`] stays the contract it is — an item carrying both
/// `image` and `video` is an image, exactly as the reference's if-chain decides.
fn media_kind(item: &Value, tag: &str, keys: &[&str]) -> bool {
    item.get("type").and_then(Value::as_str) == Some(tag)
        || keys.iter().any(|k| item.get(*k).is_some())
}

/// Which vision placeholder a part carries, and the four things the template varies by it: the
/// counter it advances, the label, the pad token, and the refusal a SYSTEM turn raises for it.
///
/// **A two-variant enum rather than the `image: bool` this file first carried, and the same for
/// [`Counting`] and [`Turn`]** (review 2026-09-01). Not house style for its own sake:
/// `vision_part` and `trimmed_content` each took TWO adjacent `bool`s, where a transposed pair
/// COMPILES and renders a wrong prompt — a video numbered as a picture, or the reverse
/// user-query scan advancing counters only the main loop may advance. Three distinct types make
/// that a type error, and what varies by media now travels with the media instead of being an
/// `if image` at three call depths.
#[derive(Copy, Clone)]
enum Media {
    Image,
    Video,
}

impl Media {
    /// The `Picture`/`Video` label and the pad token, which the template always moves together.
    fn label_and_pad(self) -> (&'static str, &'static str) {
        match self {
            Self::Image => ("Picture", "<|image_pad|>"),
            Self::Video => ("Video", "<|video_pad|>"),
        }
    }

    /// The refusal a system turn raises for this media — the template's two separate messages,
    /// both scored by `qwen_template.rs`'s refusal-site census against the vendored bytes.
    fn system_refusal(self) -> &'static str {
        match self {
            Self::Image => "System message cannot contain images.",
            Self::Video => "System message cannot contain videos.",
        }
    }

    fn counter(self, vision: &mut Vision) -> &mut u32 {
        match self {
            Self::Image => &mut vision.image,
            Self::Video => &mut vision.video,
        }
    }
}

/// Whether a content render ADVANCES the render-wide vision counters or only peeks — the
/// template macro's `do_vision_count`. `Peek` is the synthesised system turn and the reverse
/// user-query scan, both of which render content the output discards, so the numbering follows
/// main-loop message order and a part scanned twice is counted once.
#[derive(Copy, Clone, PartialEq, Eq)]
enum Counting {
    Advance,
    Peek,
}

/// Which content site a render is at — the macro's `is_system_content`. `System` refuses vision
/// outright with [`Media::system_refusal`]; `Body` is every other call. Not collapsible into
/// [`Counting`]: the reverse scan is `Peek` + `Body` where the system turn is `Peek` + `System`.
#[derive(Copy, Clone, PartialEq, Eq)]
enum Turn {
    System,
    Body,
}

/// One vision placeholder, counting and numbering it the way the template does. Everything that
/// differs between the two medias is asked of [`Media`]; the shape below is common to both.
fn vision_part(vision: &mut Vision, counting: Counting, media: Media) -> String {
    let counter = media.counter(vision);
    if counting == Counting::Advance {
        *counter += 1;
    }
    let n = *counter;
    let (label, pad) = media.label_and_pad();
    let id = if vision.add_id {
        format!("{label} {n}: ")
    } else {
        String::new()
    };
    format!("{id}<|vision_start|>{pad}<|vision_end|>")
}

/// `render_content` FOLLOWED BY `|trim` — a string as itself, a parts list as its concatenated
/// parts, `null` or a missing key as empty, anything else refused.
///
/// **One function rather than the template's macro-plus-filter pair, because all FOUR of its
/// call sites trim** — the system turn, the reverse user-query scan, and every message in the
/// main loop. Split in two they had the same four-parameter signature and jscpd matched them at
/// 38 tokens; merged, the trim cannot be forgotten at a fifth call site either.
///
/// The two flags are the macro's own parameters — `do_vision_count` and `is_system_content` —
/// and they cannot be collapsed: a system turn refuses vision outright while the reverse
/// user-query scan renders it without counting. [`Counting`] and [`Turn`] rather than two
/// `bool`s for the reason at [`Media`]: adjacent and same-typed, a swapped pair compiles.
fn trimmed_content(
    message: &Value,
    vision: &mut Vision,
    counting: Counting,
    turn: Turn,
) -> Result<String> {
    // A `String`, not a `Result<String>`: every `bail!` here is an early return, which is
    // exactly what the template's `raise_exception` is, so wrapping the match in a `Result` and
    // then `?`-ing it would be a hop no arm can reach.
    let rendered: String = match message.get("content") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => render_parts(parts, vision, counting, turn)?,
        None | Some(Value::Null) => String::new(),
        Some(_) => bail!("Unexpected content type."),
    };
    Ok(jinja_trim(&rendered).to_string())
}

/// The array form of `content`: each part rendered and concatenated, media routed through
/// `vision`.
///
/// Lifted out of [`trimmed_content`] because that match was four levels deep — function, match,
/// arm, loop, inner match — and the nesting is what the reviewer scores, not the length. The
/// `bail!`s now return from here and the one `?` in the caller propagates the identical error,
/// so the refusal text and its position in the stream are unchanged.
fn render_parts(
    parts: &[Value],
    vision: &mut Vision,
    counting: Counting,
    turn: Turn,
) -> Result<String> {
    let mut out = String::new();
    for item in parts {
        match Part::of(item) {
            Part::Media(m) if turn == Turn::System => bail!("{}", m.system_refusal()),
            Part::Media(m) => out.push_str(&vision_part(vision, counting, m)),
            // `item.text` on a part whose `text` key holds a non-string renders that
            // value through Jinja's `{{ }}`; only the string case is pinned, and a
            // non-string `text` is a client error either way.
            Part::Text => out.push_str(item.get("text").and_then(Value::as_str).unwrap_or("")),
            Part::Unexpected => bail!("Unexpected item type in content."),
        }
    }
    Ok(out)
}

/// The reasoning-effort instruction the synthesised system turn carries, or `""`.
///
/// **`medium` is the value that emits NOTHING**, and it is not the default: `xhigh` is. So the
/// three legal values produce three different prompts and only two of them carry a sentence,
/// which is worth stating because "the middle setting is the quiet one" is not guessable.
///
/// The whole block is gated on thinking being on. With `enable_thinking` strictly `false` the
/// effort is never resolved and never validated — so `enable_thinking: false` plus a bogus
/// `reasoning_effort` does NOT refuse, which is the template's behaviour and is pinned.
fn reasoning_instructions(opts: &QwenChatOpts) -> Result<&'static str> {
    if !opts.enable_thinking.undefined_or_true() {
        return Ok("");
    }
    // Jinja's `default` filter replaces UNDEFINED only, so a JSON `null` reaches the
    // membership test as `None` and is refused with `None` in the message.
    let effort = match opts.reasoning_effort {
        None => "xhigh",
        Some(Value::String(s)) => s.as_str(),
        Some(other) => bail!("{}", unexpected_effort(&python_str(other))),
    };
    match effort {
        "xhigh" => Ok(EFFORT_XHIGH),
        "low" => Ok(EFFORT_LOW),
        "medium" => Ok(""),
        other => bail!("{}", unexpected_effort(other)),
    }
}

/// The template's own refusal text, spelled once because two arms raise it.
fn unexpected_effort(value: &str) -> String {
    format!(
        "Unexpected reasoning effort {value}. Supported types are xhigh (default), medium, and \
         low."
    )
}

/// Jinja's `~` on a non-string, which is Python `str()`. Approximate for containers by this
/// module's own admission — see the header. Only ever reaches a refusal message.
///
/// **Asked of `as_bool` rather than matched on the variants**, because Python's `str()` differs
/// from its own `json.dumps` on exactly the three singletons and agrees with it on the digits of
/// every number — so JSON carries everything else and there is nothing for a `Number` arm to do.
/// (`python_json` is this crate's `json.dumps`, with the float gap the header names; a number in
/// a refusal message is the least consequential place that gap can land.) Which is also
/// why the variant form was a 24-token jscpd clone against [`argument_value`]: two matches over
/// `Value` both tailing into `python_json` are the same five tokens twice.
fn python_str(v: &Value) -> String {
    match v.as_bool() {
        Some(true) => "True".into(),
        Some(false) => "False".into(),
        None if v.is_null() => "None".into(),
        None => tokenizer::python_json(v),
    }
}

/// The system turn, in both of the template's mutually exclusive forms.
///
/// **The tools form and the plain form are one function because the CONTENT rule is shared and
/// the difference is only what sits between the header and the content.** They are also not
/// symmetric, which is why the branches are spelled out: with tools the header is always
/// emitted and the system content is APPENDED after the protocol block separated by `\n\n`;
/// without tools an empty system content and no instructions emit **nothing at all**.
fn system_turn(messages: &[Value], opts: &QwenChatOpts, vision: &mut Vision) -> Result<String> {
    let instructions = reasoning_instructions(opts)?;
    // `messages[0].role == 'system'` — the FIRST message only. A system turn anywhere else is
    // invisible here and refused by the main loop.
    let first_is_system = messages
        .first()
        .and_then(|m| m.get("role"))
        .and_then(Value::as_str)
        == Some("system");
    let content = match (first_is_system, messages.first()) {
        (true, Some(m)) => trimmed_content(m, vision, Counting::Peek, Turn::System)?,
        _ => String::new(),
    };
    let Some(lines) = tool_json_lines(opts.tools) else {
        return Ok(plain_system_turn(instructions, &content));
    };
    let mut s = format!("{IM_START}system\n");
    if !instructions.is_empty() {
        s.push_str(instructions);
        s.push_str("\n\n");
    }
    s.push_str(TOOLS_PREAMBLE);
    for line in lines {
        s.push('\n');
        s.push_str(&line);
    }
    s.push_str("\n</tools>");
    s.push_str(TOOLS_FORMAT);
    if !content.is_empty() {
        s.push_str("\n\n");
        s.push_str(&content);
    }
    s.push_str(IM_END);
    s.push('\n');
    Ok(s)
}

/// The no-tools system turn: content and instructions, content only, instructions only, or
/// nothing.
///
/// **`""` is a real outcome and not an error.** No system message and no instructions — which
/// is what `enable_thinking: false` with no system turn gives — emits no system turn at all,
/// and a prompt that opened one anyway would be one turn long before the user said anything.
fn plain_system_turn(instructions: &str, content: &str) -> String {
    match (content.is_empty(), instructions.is_empty()) {
        (true, true) => String::new(),
        (true, false) => format!("{IM_START}system\n{instructions}{IM_END}\n"),
        (false, true) => format!("{IM_START}system\n{content}{IM_END}\n"),
        (false, false) => format!("{IM_START}system\n{instructions}\n\n{content}{IM_END}\n"),
    }
}

/// The index of the last `user` message that is a real query, or `None` when there is none —
/// which is the template's `raise_exception('No user query found in messages.')`.
///
/// **A user turn whose whole trimmed content is one `<tool_response>…</tool_response>` wrapper
/// does not count as a query.** That is how the template distinguishes a tool-result turn
/// (which it frames as `user`) from something a human asked, and it is what makes
/// `preserve_thinking: false` keep the think block on the assistant turn that is still mid
/// tool loop rather than dropping it.
///
/// Walks backwards and stops at the first hit, exactly as the template's `ns.multi_step_tool`
/// latch does — so a MALFORMED user message earlier in the conversation is never rendered here
/// and never refuses. Error ordering is part of the port.
fn last_query_index(messages: &[Value], vision: &mut Vision) -> Result<Option<usize>> {
    for (i, message) in messages.iter().enumerate().rev() {
        if message.get("role").and_then(Value::as_str) != Some("user") {
            continue;
        }
        let content = trimmed_content(message, vision, Counting::Peek, Turn::Body)?;
        if !(content.starts_with("<tool_response>") && content.ends_with("</tool_response>")) {
            return Ok(Some(i));
        }
    }
    Ok(None)
}

/// An `assistant` turn: the think block (or not), the content, then any calls.
///
/// `replay_thinking` is the caller's because it is a fact about this turn's POSITION —
/// `preserve_thinking is undefined or is true or loop.index0 > last_query_index` — not about
/// the message.
///
/// **It stays a `bool` where the content path's three flags became enums** — reviewed and
/// declined 2026-09-01. [`Media`]'s argument is transposition, and each of those sat beside a
/// same-typed `bool` in the same call; this one is alone in its signature and is a predicate the
/// caller already computes and names, so an enum buys the name `replay` already carries and
/// costs a second spelling of the test at the call site. Same for `QwenChatOpts`'s two bools,
/// which mirror kwargs the caller passes as Python booleans.
fn assistant_turn(message: &Value, content: &str, replay_thinking: bool) -> String {
    let reasoning = message
        .get("reasoning_content")
        .and_then(Value::as_str)
        .map_or("", jinja_trim);
    // The block is emitted even when the reasoning is empty, as `<think>\n\n</think>\n\n` —
    // the template concatenates unconditionally inside the branch. A port that skipped it for
    // an empty string would frame every history turn differently from the way the model was
    // trained to see its own.
    let mut s = if replay_thinking {
        format!("{IM_START}assistant\n{THINK_OPEN}\n{reasoning}\n{THINK_CLOSE}\n\n{content}")
    } else {
        format!("{IM_START}assistant\n{content}")
    };
    let calls = tool_calls(message);
    for (j, call) in calls.iter().enumerate() {
        let lead = match (j == 0, content.is_empty()) {
            (true, true) => "",
            (true, false) => "\n\n",
            (false, _) => "\n",
        };
        s.push_str(&tool_call_block(call, lead));
    }
    s.push_str(IM_END);
    s.push('\n');
    s
}

/// A `tool` result turn, which the template frames as part of a **`user`** turn.
///
/// Consecutive tool results share ONE `<|im_start|>user` block: the opener appears only when
/// the previous message was not a tool, and the closer only when the next one is not. A
/// conversation that OPENS with a tool message gets no opener at all — `loop.previtem` is
/// undefined on the first iteration, which is falsy — and that is reachable, because the
/// user-query scan only needs a real user turn to exist somewhere.
fn tool_turn(messages: &[Value], i: usize, content: &str) -> String {
    let role_at = |j: usize| {
        messages
            .get(j)
            .and_then(|m| m.get("role"))
            .and_then(Value::as_str)
    };
    let mut s = String::new();
    if i > 0 && role_at(i - 1) != Some("tool") {
        s.push_str(IM_START);
        s.push_str("user");
    }
    s.push_str("\n<tool_response>\n");
    s.push_str(content);
    s.push_str("\n</tool_response>");
    if role_at(i + 1) != Some("tool") {
        s.push_str(IM_END);
        s.push('\n');
    }
    s
}

/// The generation prompt: the assistant header plus a think block that is OPEN in thinking mode
/// and a CLOSED EMPTY one otherwise.
///
/// **The prompt always ends inside or just past a `<think>` block**, which is the contract
/// `serve::oai::split_think` reads back: with thinking on the model closes the block itself and
/// everything after `</think>` is the answer; with thinking off `</think>` is already in the
/// PROMPT and the generation carries no tags at all.
fn generation_prompt(opts: &QwenChatOpts) -> String {
    let think = if opts.enable_thinking.defined_and_false() {
        format!("{THINK_OPEN}\n\n{THINK_CLOSE}\n\n")
    } else {
        format!("{THINK_OPEN}\n")
    };
    format!("{IM_START}assistant\n{think}")
}

/// Render an OpenAI-shaped `messages` array as Qwen3.8-Flash-Next's chat framing.
///
/// **Messages are `Value`, not a typed model**, for [`crate::glimmer_encoding::render`]'s
/// reason: the template branches on optional fields most callers never set, and the pinned
/// cases are themselves JSON.
///
/// **This returns `Result`, and that is the template's doing rather than a house preference.**
/// `chat_template.jinja` calls `raise_exception` in NINE places — a count `qwen_template.rs`
/// gates, because six files here first said eight and nothing recomputed it — and three are
/// reachable from an ordinary client: a `developer` role (which OpenAI clients send) is
/// `Unexpected message role.`, a conversation with no real user turn is `No user query found in
/// messages.`, and a system turn that is not first is `System message must be at the
/// beginning.`. Glimmer's port returns a bare `String` because its template silently drops an
/// unknown role; this one cannot, and a renderer that invented a framing for a case the
/// reference refuses would be producing a prompt the model has never seen.
///
/// The refusal STRINGS are the template's own, verbatim, so the pin can score them.
pub fn render(messages: &[Value], opts: &QwenChatOpts) -> Result<String> {
    if messages.is_empty() {
        bail!("No messages provided.");
    }
    let vision = &mut Vision {
        image: 0,
        video: 0,
        add_id: opts.add_vision_id,
    };
    let mut s = system_turn(messages, opts, vision)?;
    let Some(last_query) = last_query_index(messages, vision)? else {
        bail!("No user query found in messages.");
    };
    for (i, message) in messages.iter().enumerate() {
        // Rendered BEFORE the role is checked, exactly as the template does — so a malformed
        // content part refuses ahead of a misplaced system turn, and the vision counters
        // advance for a message whose output is discarded.
        let content = trimmed_content(message, vision, Counting::Advance, Turn::Body)?;
        match message.get("role").and_then(Value::as_str) {
            Some("system") => {
                if i != 0 {
                    bail!("System message must be at the beginning.");
                }
                // Already rendered into the preamble; the main loop emits nothing for it.
            }
            Some("user") => s.push_str(&format!("{IM_START}user\n{content}{IM_END}\n")),
            Some("assistant") => {
                let replay = opts.preserve_thinking.undefined_or_true() || i > last_query;
                s.push_str(&assistant_turn(message, &content, replay));
            }
            Some("tool") => s.push_str(&tool_turn(messages, i, &content)),
            _ => bail!("Unexpected message role."),
        }
    }
    if opts.add_generation_prompt {
        s.push_str(&generation_prompt(opts));
    }
    Ok(s)
}
