//! Tokenizer: loads the snapshot's `tokenizer.json` (HuggingFace format) via
//! the `tokenizers` crate. Hand-rolling BPE would be the archetypal wheel to
//! not reinvent; this is a 19 MB trained vocab.
//!
//! Ported from `old:src/artifact/tokenizer.rs` (`wt/glimmer-s2` @ 6b7f496), **GLM surface
//! only**, with the bodies and their comments travelling verbatim — in this repo a comment
//! carries the measurement that justified the choice, so a re-worded one loses evidence.
//! `crates/artifact/tests/glm_chat_parity.rs` pins the ids this port emits against the ids
//! the reference emits, which is the only thing that can catch the drift the
//! `encode_chat_turns` doc records below.
//!
//! **The seam this port cuts, and who owns the other side.** `encode_dsv4`, `Message` and
//! `EncodeOpts` are DeepSeek-V4's, backed by `old:src/artifact/dsv4_encoding.rs` (2822
//! lines) — **deferred to M8**, which owns the V4 path. The cut is clean because the two
//! encoders share no framing: GLM builds a token-ID list and V4 builds a *string* that is
//! then tokenized, and the old tree's own module header says of the pair that "the two must
//! not converge". What crosses the seam is one helper: [`python_json`], which is here
//! because GLM's [`tools_system_turn`] and [`tool_call_markup`] need it, and which V4's port
//! should reach for rather than write again.
//!
//! > **UPDATED 2026-08-16. M8 LANDED and the seam held.** [`crate::v4_encoding`] is the other
//! > side, split across four files because the reference module is 2822 lines against this
//! > tree's 800-line cap. It reaches for [`python_json`] and [`json_truthy`] exactly as
//! > predicted and writes neither again — so the pair below now has THREE callers in three
//! > modules and `pub(crate)` is load-bearing twice over.
//! >
//! > **`encode_dsv4` itself did NOT come with it, deliberately.** In the reference it is one
//! > line on this type — `self.encode(&dsv4_encoding::encode_messages(messages, opts)?)` — and
//! > its only caller is `main.rs`'s V4 `-bench` branch, which this tree does not have yet.
//! > Adding it now would put a `pub fn` with no caller on [`Tokenizer`], which is the shape
//! > this header's own `json_truthy` paragraph and `encode_chat_continuation`'s deletion note
//! > both argue against. It lands with the V4 engine arm, beside the caller that needs it;
//! > until then `crates/artifact/tests/v4_encoding_gold.rs` exercises the same composition
//! > (`Tokenizer::encode` over `encode_messages`'s string) and pins that every framing token
//! > survives it as ONE id, which is the only property the wrapper would have added.
//! >
//! > **UPDATED: the V4 ENGINE ARM landed and brought its caller.** [`Tokenizer::encode_dsv4`]
//! > is below, one line, with `main.rs`'s V4 `--bench` branch as the caller the paragraph
//! > above said it would wait for. The composition test stays: it pins the property the
//! > wrapper does not add, which is that every framing token survives tokenization as one id.
//!
//! `json_truthy` did NOT come with it, deliberately: in the reference its only callers are
//! `dsv4_encoding` and `glimmer_encoding`, so porting it now would land a function with no
//! caller — and `warnings = deny` makes that a build error, which is the correct pressure.
//! It moves with whichever of those two ports lands first, and *beside* `python_json`, for
//! the reason the reference records at its declaration: two ports getting Python's truth
//! table right independently is not something to have twice.
//!
//! > **UPDATED 2026-08-16. It has moved — M7 (`glimmer_encoding`) landed first.**
//! > [`json_truthy`] is below, beside [`python_json`], with one caller. The paragraph above is
//! > kept rather than rewritten because it records why the item was *withheld*, which is the
//! > half a reader cannot reconstruct from the code being present.
//! >
//! > **UPDATED AGAIN 2026-08-16, same day: M8 brought the second caller.**
//! > [`crate::v4_encoding::message`] tests `wo_eos` and a stray `tools` key with it and
//! > [`crate::v4_encoding::render`] tests `response_format` — which is the `if response_format:`
//! > this function's own doc was written against. The prediction that two ports would otherwise
//! > get Python's truth table right independently is now a fact with three witnesses.

use anyhow::{Context, Result, anyhow, bail};
use serde_json::Value;
use std::io::{self, Write};

/// Announce a survivable defect in the artifact's tokenizer files.
///
/// **`eprintln!`, not `tracing::warn!`, and that is a deliberate difference from the
/// reference.** `rivoli-artifact` is host-only and carries no `tracing` edge; more to the
/// point, the crate's consumers include the converter binaries under `crates/cli/src/bin`,
/// none of which installs a subscriber — a `warn!` there emits *nothing*, which is the
/// silent-failure shape every warning below exists to avoid. `format/meta.rs` and
/// `quant/naming.rs` already announce this way for the same reason.
fn warn(msg: &str) {
    eprintln!("tokenizer: {msg}");
}

pub struct Tokenizer {
    vocab: Vocabulary,
    /// End-of-sequence ids from generation_config (any one stops decode).
    pub eos: Vec<u32>,
}

/// Which vocabulary format this artifact carries.
///
/// **An enum rather than a trait object**, because the two arms are not interchangeable and
/// pretending otherwise is the bug: only the HuggingFace arm can answer `token_to_id`, which is
/// what every chat-framing method here is built on. A trait would have to include those methods
/// and the tiktoken arm would implement them by returning `None`, turning "this model has no
/// chat framing" into "this model's framing tokens are all missing" — which `encode_chat_turns`
/// already handles by silently encoding RAW. Refusing loudly is the whole point, so the
/// distinction stays visible in the type.
///
/// **Both arms are boxed, not just the larger one.** `tokenizers::Tokenizer` is ~1480 bytes and
/// `tiktoken::Vocab` ~368, so boxing only the first merely inverts which variant clippy's
/// `large_enum_variant` complains about (measured, in that order, 2026-08-17). One `Tokenizer`
/// exists per process and lives for its whole life, so two indirections cost two allocations at
/// startup and keep this enum at a pointer.
enum Vocabulary {
    /// GLM, DeepSeek-V4 and Muse Glimmer: `tokenizer.json` through the `tokenizers` crate.
    Hf(Box<tokenizers::Tokenizer>),
    /// Kimi-K3: `tiktoken.model` plus a positional special block — see [`crate::tiktoken`].
    Tiktoken(Box<crate::tiktoken::Vocab>),
}

/// The snapshot's stop tokens, as the engine reads them: `generation_config.json`'s
/// `eos_token_id`, either an array of ints or a bare one.
///
/// **`pub` and free rather than private to [`Tokenizer`], because a second caller needs to read
/// this file the SAME way.** `bin/convert_glimmer` refuses a checkpoint whose ids are unusable —
/// Muse Glimmer carries two (`[200001, 200008]`) and an artifact with none cannot terminate a
/// decode at all. That converter had its own copy of this parse for one commit, with a comment
/// asserting the two "match"; a claim a shared function makes structural instead (review,
/// 2026-08-13). A gate that reads the file differently from the engine certifies a file the engine
/// then rejects, or passes one it cannot use.
///
/// > **PORT NOTE 2026-08-16. That second caller does not exist in this tree yet**, so the
/// > paragraph above is the reference's reason, not a description of today. The rewrite's
/// > converter (`crates/cli/src/bin/convert.rs`) refuses a *missing* `generation_config.json`
/// > structurally at `format::meta::finish_artifact` without parsing its ids; the Glimmer
/// > converter that reads them lands with M7. The item stays `pub` and free because that is
/// > the shape the second caller needs, and moving it later is the change that invites the
/// > second copy.
/// >
/// > **UPDATED 2026-08-16, same day: the anticipated shape arrived.**
/// > `crates/cli/src/bin/convert_glimmer.rs::eos_ids` is the second caller, and it is exactly
/// > the one described — it wraps this in a non-empty `ensure!` and a vocabulary-bound check,
/// > because a Glimmer artifact whose only stop tokens are unusable cannot terminate a decode
/// > at all. It calls this rather than re-parsing the file, so the gate and the engine read
/// > `eos_token_id` the same way by construction rather than by a comment claiming they
/// > "match". The `pub`-and-free shape is now load-bearing rather than anticipatory.
///
/// **A MISSING key is `Ok(vec![])`, not an error.** It is the same outcome as an empty array — no
/// stop tokens — and both callers act on the outcome rather than on how it was spelled. Only an
/// unreadable or non-JSON file is an `Err`, and each says which. (It was `Err` until 2026-08-13,
/// which made `Tokenizer::load` report a missing key as "could not read eos ids" and would have
/// given the converter two messages for one condition.)
pub fn eos_token_ids(dir: &str) -> Result<Vec<u32>> {
    let p = format!("{dir}/generation_config.json");
    let text = std::fs::read_to_string(&p).with_context(|| format!("read {p}"))?;
    let v: Value = serde_json::from_str(&text).with_context(|| format!("{p} is not JSON"))?;
    Ok(match v.get("eos_token_id") {
        Some(Value::Array(a)) => a
            .iter()
            .filter_map(|x| x.as_u64().map(|n| n as u32))
            .collect(),
        Some(other) => other.as_u64().map(|n| vec![n as u32]).unwrap_or_default(),
        None => Vec::new(),
    })
}

/// Thinking control for [`Tokenizer::encode_chat_turns`], named after the parameters of the
/// same names in the checkpoint's `chat_template.jinja`.
///
/// Thinking is a PREFILL, not a flag the model reads: the generation prompt ends at an open
/// `<think>` and the model reasons until it emits `</think>`, or ends at `<think></think>`
/// and it answers straight away. The vocabulary has a `/nothink` token (154851) but the
/// template never emits it, so neither does this.
#[derive(Debug, Clone, Copy, Default)]
pub struct ChatOpts<'a> {
    /// Leave `<think>` open so the model reasons first.
    ///
    /// **The template's default is `true`; this struct's is `false`, deliberately.** At
    /// ~2.7 tok/s a reasoning block is tens of seconds of silence before the first word of
    /// the answer, and an OpenAI client has no way to ask for it to stop. `--think` and the
    /// request's `enable_thinking` turn it back on per server and per request.
    pub thinking: bool,
    /// `Some("high")` renders "Reasoning Effort: High"; everything else renders "Max",
    /// which is the template's own `capitalize` of its default. Ignored unless `thinking`.
    pub reasoning_effort: Option<&'a str>,
    /// OpenAI `tools` — the raw array, each entry either `{"type","function":{...}}` or a
    /// bare function object. Renders the template's `# Tools` system turn, which is the
    /// ONLY thing that teaches the model this checkpoint's `<tool_call>` syntax; invent a
    /// preamble instead and it reaches for other frameworks' conventions.
    pub tools: Option<&'a Value>,
}

/// `serde_json`, but with Python's default separators — `", "` between items and `": "`
/// after a key.
///
/// Not cosmetic. The template renders tool schemas through Jinja's `tojson`, which is
/// `json.dumps`, which spaces them that way — so that is the byte sequence the model was
/// trained on, and `serde_json`'s compact form tokenizes differently. `ensure_ascii=False`
/// needs no handling: serde_json never escapes non-ASCII, which is the same thing.
struct PythonSpacing;

/// Python's `item_separator` — `", "` before every element but the first.
///
/// ONE function, not one per hook: `json.dumps` uses the same separator between array
/// elements and between an object's members, so two spellings could only ever differ by
/// being wrong. The hooks below are thin because this is the whole rule.
fn item_sep<W: ?Sized + Write>(w: &mut W, first: bool) -> io::Result<()> {
    if first { Ok(()) } else { w.write_all(b", ") }
}

impl serde_json::ser::Formatter for PythonSpacing {
    fn begin_array_value<W: ?Sized + Write>(&mut self, w: &mut W, first: bool) -> io::Result<()> {
        item_sep(w, first)
    }
    fn begin_object_key<W: ?Sized + Write>(&mut self, w: &mut W, first: bool) -> io::Result<()> {
        item_sep(w, first)
    }
    fn begin_object_value<W: ?Sized + Write>(&mut self, w: &mut W) -> io::Result<()> {
        w.write_all(b": ")
    }
}

/// `pub(crate)` because `crate::v4_encoding` needs the identical spacing for DeepSeek's
/// tool schemas — its reference implementation renders them with the same `json.dumps`. A
/// second copy would be a `build.rs` duplication error, and worse, a second thing to fix.
/// (Named `artifact::dsv4_encoding` here until M8, which renamed the module `v4_encoding` for
/// the reason its header gives; the path is corrected rather than the claim.)
///
/// > **PORT NOTE 2026-08-16.** That consumer arrives with M8 and the Glimmer one with M7;
/// > until then the only caller is this module. It is `pub(crate)` now for the reason above
/// > and not on speculation — the alternative is a private helper that the next port has no
/// > way to see, and the old tree measured exactly that outcome (Glimmer wrote a second copy
/// > of its neighbour `json_truthy`, and the duplication gate reported it on the first
/// > compile, 2026-08-14).
/// >
/// > **UPDATED 2026-08-16, same day: M7 landed and the speculation is now a fact.**
/// > [`crate::glimmer_encoding`] renders Muse Glimmer's ATEM tool schemas through this, so
/// > there are two callers in two modules and the `pub(crate)` is load-bearing rather than
/// > anticipatory.
/// >
/// > **UPDATED AGAIN 2026-08-16: the consumer this comment was WRITTEN for arrived.**
/// > `crate::v4_encoding::render` renders DeepSeek's tool schemas and its `## Response Format:`
/// > block through this, and `v4_encoding::tests::boundary`'s
/// > `numeric_rendering_diverges_from_python` pins the one thing it does NOT reproduce —
/// > Python's number repr — as a measured table rather than an unstated gap. That test is the
/// > only gate on this function's limits; see `v4_encoding`'s header for why the complete fix
/// > (`serde_json/arbitrary_precision`) is crate-wide and is recorded rather than half-done.
pub(crate) fn python_json(v: &Value) -> String {
    let mut buf = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut buf, PythonSpacing);
    match serde::Serialize::serialize(v, &mut ser) {
        Ok(()) => String::from_utf8(buf).unwrap_or_default(),
        Err(_) => v.to_string(),
    }
}

/// Python truthiness, which is what a bare `{%- if x -%}` in a Jinja chat template tests and
/// what `if response_format:` tests in DeepSeek's Python one.
///
/// `false`, `0`, `0.0`, `""`, `[]`, `{}` and `null` are false; everything else — including the
/// STRING `"false"` and the string `"0"` — is true.
///
/// **Beside [`python_json`] because it is the same kind of thing: a Python semantic this crate
/// has to reproduce exactly, owned by neither model.** In the old tree it began as
/// `dsv4_encoding::json_truthy`; Glimmer's port needed it for `{%- if tools -%}` and
/// `end_turn`, wrote a second copy, and `build.rs` reported the clone on the first compile
/// (2026-08-14). Two ports getting Python's truth table right independently is not something
/// to have twice.
///
/// > **PORT NOTE 2026-08-16.** This module's header said this item "moves with whichever of
/// > those two ports lands first, and *beside* `python_json`". **M7 landed first**, so it is
/// > here, and `glimmer_encoding` is its one caller until M8 brings V4's. The header has been
/// > updated in place rather than left describing a decision that has been made.
pub(crate) fn json_truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// One call the model made, as the template writes it back into the conversation:
/// `<tool_call>{name}<arg_key>{k}</arg_key><arg_value>{v}</arg_value>...</tool_call>`.
///
/// A STRING argument is emitted raw and everything else as JSON — the template's
/// `v | tojson if v is not string else v`, which is also what makes the parse in
/// `serve::parse_tool_calls` round-trip.
pub fn tool_call_markup(name: &str, arguments: &Value) -> String {
    let mut s = format!("<tool_call>{name}");
    if let Some(args) = arguments.as_object() {
        for (k, v) in args {
            let rendered = match v {
                Value::String(t) => t.clone(),
                other => python_json(other),
            };
            s.push_str(&format!(
                "<arg_key>{k}</arg_key><arg_value>{rendered}</arg_value>"
            ));
        }
    }
    s.push_str("</tool_call>");
    s
}

/// A tool RESULT, as the template frames it inside an `<|observation|>` turn. Consecutive
/// results share one observation turn — the caller concatenates these, see
/// `serve::messages_to_turns`.
pub fn tool_response_markup(content: &str) -> String {
    format!("<tool_response>{content}</tool_response>")
}

/// The `# Tools` system turn, byte-for-byte from `chat_template.jinja`.
///
/// The Jinja is rendered with `trim_blocks`/`lstrip_blocks` (what transformers sets), so the
/// block tags around the loop contribute nothing and each schema lands on its own line.
/// `defer_loading` and `strict` are dropped per the template's own macro.
fn tools_system_turn(tools: &Value) -> String {
    let mut s = String::from(
        "\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n\
         You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n",
    );
    for tool in tools.as_array().into_iter().flatten() {
        let f = tool.get("function").unwrap_or(tool);
        let cleaned: serde_json::Map<_, _> = f
            .as_object()
            .into_iter()
            .flatten()
            .filter(|(k, _)| k.as_str() != "defer_loading" && k.as_str() != "strict")
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        s.push_str(&python_json(&Value::Object(cleaned)));
        s.push('\n');
    }
    s.push_str(
        "</tools>\n\nFor each function call, output the function name and arguments within \
         the following XML format:\n<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key>\
         <arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}</arg_key>\
         <arg_value>{arg-value-2}</arg_value>...</tool_call>",
    );
    s
}

/// The six turn tokens [`Tokenizer::encode_chat_turns`] cannot frame a conversation without,
/// resolved once per call. Named fields rather than the reference's positional `sp[i]`
/// indexing: the two `<think>` ids are adjacent and interchangeable by type, and the port
/// is the moment that swap would have been easiest to make and hardest to see.
struct ChatTokens {
    gmask: u32,
    sop: u32,
    user: u32,
    assistant: u32,
    think_open: u32,
    think_close: u32,
}

impl ChatTokens {
    /// `None` when ANY of the six is missing — an artifact whose tokenizer predates the chat
    /// tokens. `<think>`/`</think>` are in the same lookup because the generation prompt
    /// cannot be built without them.
    fn resolve(inner: &tokenizers::Tokenizer) -> Option<Self> {
        let id = |t: &str| inner.token_to_id(t);
        Some(Self {
            gmask: id("[gMASK]")?,
            sop: id("<sop>")?,
            user: id("<|user|>")?,
            assistant: id("<|assistant|>")?,
            think_open: id("<think>")?,
            think_close: id("</think>")?,
        })
    }
}

impl Tokenizer {
    /// Load whichever vocabulary the artifact carries.
    ///
    /// **Sniffed by FILE, not by architecture**, so this stays the one door every arm enters
    /// through and `main` needs no per-arch branch before it. `tokenizer.json` wins when both
    /// exist, which is the conservative order: it is what the other three checkpoints ship and
    /// what every recorded benchmark id was produced with.
    ///
    /// > **Kimi-K3 could not be opened AT ALL before this landed** (2026-08-17). This function
    /// > ran unconditionally before the architecture match, took `tokenizer.json` as the only
    /// > possibility, and K3 ships none — so `--bench` failed on a 1.42 TiB artifact that was
    /// > otherwise complete and verified. `docs/investigations/k3-first-checkpoint.md` §4.
    pub fn load(snapshot_dir: &str) -> Result<Self> {
        let hf = format!("{snapshot_dir}/tokenizer.json");
        let tk = format!("{snapshot_dir}/tiktoken.model");
        let vocab = if std::fs::metadata(&hf).is_ok() {
            Vocabulary::Hf(Box::new(
                tokenizers::Tokenizer::from_file(&hf).map_err(|e| anyhow!("load {hf}: {e}"))?,
            ))
        } else if std::fs::metadata(&tk).is_ok() {
            Vocabulary::Tiktoken(Box::new(crate::tiktoken::Vocab::load(snapshot_dir)?))
        } else {
            // Both names, because which one is missing depends on the model and a reader
            // holding the artifact needs to know which they should have.
            bail!(
                "{snapshot_dir} has neither tokenizer.json (HuggingFace: GLM, DeepSeek-V4, \
                 Muse Glimmer) nor tiktoken.model (Kimi-K3) — an artifact carries its own \
                 tokenizer, so one of the two must be there"
            );
        };
        // An empty eos means decode can never stop on an end token (runaway
        // generation) — surface a broken generation_config loudly, don't swallow.
        let eos = match eos_token_ids(snapshot_dir) {
            Ok(e) if !e.is_empty() => e,
            Ok(_) => {
                warn("generation_config has no eos_token_id — decode won't stop on EOS");
                Vec::new()
            }
            Err(e) => {
                warn(&format!(
                    "could not read eos ids ({e}) — decode won't stop on EOS"
                ));
                Vec::new()
            }
        };
        Ok(Self { vocab, eos })
    }

    /// The HuggingFace vocabulary, or a refusal naming why the caller cannot have it.
    ///
    /// Every chat-framing method below is written against `token_to_id`, which only this arm
    /// has. Refusing rather than falling back is deliberate — [`Vocabulary`]'s doc carries why.
    ///
    /// > **A second accessor `hf_opt() -> Option<..>` stood here and is deleted** (review,
    /// > 2026-08-17). It existed only because `turn_header` returns `u32` and cannot use `?`;
    /// > the doc block above had landed on IT rather than on this function, leaving `hf` with no
    /// > doc at all. Threading the already-resolved `&Tokenizer` into `turn_header` removes the
    /// > second accessor, the `Option` that could never be `None` there, and the paragraph
    /// > explaining which of two near-identical accessors to reach for.
    fn hf(&self, what: &str) -> Result<&tokenizers::Tokenizer> {
        match &self.vocab {
            Vocabulary::Hf(t) => Ok(t),
            Vocabulary::Tiktoken(_) => bail!(
                "{what} needs a HuggingFace tokenizer's token_to_id, and this artifact carries \
                 a tiktoken vocabulary (Kimi-K3). K3's chat framing is its first-party XTML \
                 encoder and is not ported; --bench encodes RAW on that architecture"
            ),
        }
    }

    /// GLM turn framing: `[gMASK] <sop> <|user|> {text} <|assistant|> <think> </think>`.
    ///
    /// > **CORRECTED IN THE PORT, 2026-08-16.** The reference's copy of this line still
    /// > reads `<|user|> \n {text} <|assistant|> \n` — the GLM-4 framing that
    /// > [`Self::encode_chat_turns`]'s own dated note retracts, and that its code stopped
    /// > emitting on 2026-08-01. The *code* was fixed then and the summary line was not, so
    /// > the reference documents the defect it fixed. Carried over corrected rather than
    /// > verbatim, and pinned by ids: `crates/artifact/tests/glm_chat_parity.rs` shows
    /// > `encode_chat("Hi")` = `[gMASK] <sop> <|user|> Hi <|assistant|> <think> </think>`
    /// > with no newline token anywhere in it.
    ///
    /// **Without this the model can never stop.** Two of its three EOS ids are turn
    /// boundaries (`<|user|>` 154827, `<|observation|>` 154829) and the third is
    /// `<|endoftext|>` — all of which an instruct model emits at the end of an ASSISTANT
    /// TURN. Fed raw text it is doing document continuation, is never in a turn, and has
    /// no reason to emit any of them: across 56 benchmark runs not one terminated
    /// naturally, every one ran to its token limit. Forced to keep writing past the end
    /// of its answer it drifts into list scaffolding and then loops
    /// (`**Memory Product.**` x329) — which is what invalidated every matrix cell above
    /// 2048 tokens. See docs/measurement/benchmarks.md's retraction.
    ///
    /// Built from token IDS rather than by encoding the literal template text, so it does
    /// not depend on the tokenizer choosing to match `[gMASK]` as a special token inside a
    /// string. No Jinja engine either — see [`Tokenizer::encode_chat_turns`], which is a
    /// hand-port of the checkpoint's `chat_template.jinja` down to the byte.
    pub fn encode_chat(&self, text: &str) -> Result<Vec<u32>> {
        self.encode_chat_turns(&[("user", text)], &ChatOpts::default())
    }

    // `encode_chat_continuation` (the reference's mid-conversation follow-up framing) is
    // NOT ported: its only caller is `-bench`'s follow-up script, which this tree does not
    // have. Same rule the header applies to `json_truthy` — a function with no caller is
    // dead surface `warnings = deny` cannot see behind `pub`. It is `encode_chat` minus
    // the two-token `[gMASK] <sop>` prefix; re-derive it WITH its caller.

    /// The system turns the template emits BEFORE the conversation — reasoning effort, then
    /// the tool declarations — in that order, which is the order the model saw them in
    /// training.
    ///
    /// Split out of [`Self::encode_chat_turns`] in the port: it is the one part of the
    /// template that is optional at both ends (`<|system|>` may be absent from the vocab,
    /// and both turns are opt-in), so it is also the part whose conditions do not belong
    /// tangled with the per-turn loop.
    fn preamble(&self, opts: &ChatOpts) -> Result<Vec<u32>> {
        let Some(system) = self.hf("chat preamble")?.token_to_id("<|system|>") else {
            return Ok(Vec::new());
        };
        let mut out = Vec::new();
        // The template emits this whenever thinking is on, BEFORE the conversation, so the
        // model saw it there in training. `capitalize` of the template's own default makes
        // anything that is not "high" into "Max" — including "low" and "medium", which is
        // the template's behaviour and not a shortcut taken here.
        if opts.thinking {
            let effort = if opts.reasoning_effort == Some("high") {
                "High"
            } else {
                "Max"
            };
            out.push(system);
            out.extend_from_slice(&self.encode(&format!("Reasoning Effort: {effort}"))?);
        }
        // The tool declarations, after the effort turn and before the conversation.
        if let Some(tools) = opts
            .tools
            .filter(|t| t.as_array().is_some_and(|a| !a.is_empty()))
        {
            out.push(system);
            out.extend_from_slice(&self.encode(&tools_system_turn(tools))?);
        }
        Ok(out)
    }

    /// The turn token a role is framed with. Roles map to GLM's own turn tokens
    /// (`system`/`user`/`assistant`/`observation`); an unknown role is framed as `user`,
    /// since the alternative — dropping the turn — loses content silently.
    /// `hf` is passed in rather than re-fetched: this returns `u32` and cannot use `?`, and its
    /// only caller has already resolved the HuggingFace arm — so the tiktoken case is
    /// unreachable here by construction instead of by a second accessor that has to invent an
    /// answer for it.
    fn turn_header(&self, hf: &tokenizers::Tokenizer, role: &str, sp: &ChatTokens) -> u32 {
        match role {
            "user" => sp.user,
            "assistant" => sp.assistant,
            // Only looked up when actually used: an artifact can carry the tokens above
            // without these, and a `system` message is optional.
            other => match hf.token_to_id(&format!("<|{other}|>")) {
                Some(id) => id,
                None => {
                    warn(&format!(
                        "tokenizer has no <|{other}|> turn token — framing it as user"
                    ));
                    sp.user
                }
            },
        }
    }

    /// The same framing over a MULTI-TURN conversation, which is what an OpenAI `messages`
    /// array is. **This is a hand-port of the checkpoint's `chat_template.jinja`, and it is
    /// meant to match it byte for byte:**
    ///
    /// ```text
    /// [gMASK] <sop>
    /// [<|system|> "Reasoning Effort: High|Max"]      -- only when thinking
    /// <|role|> {content}                             -- per turn, NO separator
    /// <|assistant|> <think> </think> {content}       -- assistant turns clear reasoning
    /// <|assistant|> <think> [</think>]               -- the generation prompt
    /// ```
    ///
    /// > **CORRECTED 2026-08-01. There is no newline after the role token, and there never
    /// > was.** This function used to emit `<|role|> \n {content}` and end the prompt at
    /// > `<|assistant|> \n`, which is the GLM-4 template, not this checkpoint's. Every
    /// > benchmark before that date was measured one token off-template per turn and with
    /// > the thinking prefill missing entirely, so its text is not comparable with anything
    /// > measured after. The source of truth is `chat_template.jinja` in the fp8 checkpoint
    /// > (`manifest.json`'s `i4_source.src`); the converted artifact does not carry it,
    /// > which is why this drifted unnoticed.
    ///
    /// The `<think>` prefill is the whole of thinking control — see [`ChatOpts`]. Note the
    /// template never emits the `/nothink` token, despite the vocab having one.
    pub fn encode_chat_turns(&self, turns: &[(&str, &str)], opts: &ChatOpts) -> Result<Vec<u32>> {
        // Missing any of the six means an artifact whose tokenizer predates the chat tokens;
        // fall back rather than fail, but say so — silent raw encoding is the bug this
        // function exists to fix.
        let hf = self.hf("chat framing")?;
        let Some(sp) = ChatTokens::resolve(hf) else {
            warn(
                "tokenizer lacks the GLM chat tokens ([gMASK]/<sop>/<|user|>/<|assistant|>/\
                 <think>) — encoding the prompt RAW. Decode will not stop on EOS.",
            );
            return self.encode(&turns.iter().map(|(_, t)| *t).collect::<Vec<_>>().join("\n"));
        };
        let mut out = vec![sp.gmask, sp.sop];
        out.extend_from_slice(&self.preamble(opts)?);

        for (role, text) in turns {
            out.push(self.turn_header(hf, role, &sp));
            if *role == "assistant" {
                // History carries NO reasoning. The template keeps it only for assistant
                // turns after the last user message (a continuation), so for a request that
                // ends in a user turn — every request a chat client makes — this is all of
                // them. `.trim()` matches the template's `content.strip()`.
                out.push(sp.think_open);
                out.push(sp.think_close);
                out.extend_from_slice(&self.encode(text.trim())?);
            } else {
                out.extend_from_slice(&self.encode(text)?);
            }
        }

        // The generation prompt. An OPEN `<think>` is the template's default and makes the
        // model reason before answering; closing it immediately is how thinking is turned
        // off. Either way the model is inside the structure it was trained on.
        out.push(sp.assistant);
        out.push(sp.think_open);
        if !opts.thinking {
            out.push(sp.think_close);
        }
        Ok(out)
    }

    /// DeepSeek-V4's turn framing: [`crate::v4_encoding::encode_messages`] builds the STRING
    /// and this tokenizes it.
    ///
    /// **The two encoders share no framing, and that is why this is a delegation rather than a
    /// second [`Self::encode_chat_turns`].** GLM builds a token-ID list directly, from ids
    /// looked up by name, so it never depends on the tokenizer choosing to match `[gMASK]` as
    /// a special token inside a string. V4 builds a string with its markers written out and
    /// relies on exactly that match — which is a real dependency on the vocab, and the reason
    /// `crates/artifact/tests/v4_encoding_gold.rs` pins that every framing token survives this
    /// composition as ONE id rather than several. The old tree's own module header says of the
    /// pair that "the two must not converge"; keeping this a one-liner over a separate
    /// renderer is what honours that.
    pub fn encode_dsv4(
        &self,
        messages: Vec<crate::v4_encoding::Message>,
        opts: &crate::v4_encoding::EncodeOpts,
    ) -> Result<Vec<u32>> {
        self.encode(&crate::v4_encoding::encode_messages(messages, opts)?)
    }

    /// Encode prompt text to token ids with NO special tokens — raw continuation.
    /// Used by `--ppl`, which scores a fixed corpus rather than a chat turn. It also
    /// served `--raw-prompt`, a flag deleted 2026-08-01 for reproducing pre-templating
    /// benchmark numbers that no recorded command line ever asked for.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        match &self.vocab {
            Vocabulary::Hf(t) => {
                let enc = t.encode(text, false).map_err(|e| anyhow!("encode: {e}"))?;
                Ok(enc.get_ids().to_vec())
            }
            Vocabulary::Tiktoken(v) => v.encode(text),
        }
    }

    /// Decode a whole id sequence to text at once. Byte-level BPE splits one
    /// codepoint across several tokens, so decoding the complete sequence is
    /// what keeps multi-token characters (and any trailing partial one) intact —
    /// there is no incremental-flush footgun. Streaming detok belongs to server
    /// mode; it can be added there when it exists.
    pub fn decode_all(&self, ids: &[u32]) -> Result<String> {
        match &self.vocab {
            Vocabulary::Hf(t) => t.decode(ids, false).map_err(|e| anyhow!("decode: {e}")),
            Vocabulary::Tiktoken(v) => v.decode(ids),
        }
    }
}

#[cfg(test)]
mod tests {
    // Crate-wide `unwrap`/`expect` are `deny`; a firing one IS the report.
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use serde_json::json;

    /// The tool markup helpers need NO artifact — they are pure string production — so they
    /// are gated here rather than in `tests/glm_chat_parity.rs`, which skips without the
    /// 19.3 MB vocab. Ported ahead of their only caller (M6's serve piece), and an item with
    /// no caller and no gate is exactly the shape that rots between milestones.
    #[test]
    fn a_string_argument_is_raw_and_everything_else_is_python_spaced_json() {
        let args = json!({ "path": "/tmp/x", "depth": 2, "flags": ["a", "b"] });
        assert_eq!(
            tool_call_markup("read", &args),
            "<tool_call>read\
             <arg_key>path</arg_key><arg_value>/tmp/x</arg_value>\
             <arg_key>depth</arg_key><arg_value>2</arg_value>\
             <arg_key>flags</arg_key><arg_value>[\"a\", \"b\"]</arg_value>\
             </tool_call>"
        );
        // The string argument arrived UNQUOTED and the array with `", "` between elements.
        // `serde_json`'s compact form would have written `["a","b"]`, which tokenizes
        // differently from what the model was trained on — the whole reason `PythonSpacing`
        // exists. Both halves of that claim are in the one expectation above.
        assert_eq!(
            tool_response_markup("42"),
            "<tool_response>42</tool_response>"
        );
    }

    /// `json.dumps`'s separators, stated once against a nested value so a formatter hook
    /// that got only arrays right (or only objects) cannot pass.
    #[test]
    fn python_json_spaces_both_separators() {
        assert_eq!(
            python_json(&json!({ "a": [1, 2], "b": { "c": null } })),
            "{\"a\": [1, 2], \"b\": {\"c\": null}}"
        );
    }

    /// The `# Tools` turn drops `defer_loading` and `strict` per the template's own macro,
    /// unwraps the OpenAI `{"type","function":{…}}` envelope, and puts each schema on its
    /// own line. A tool declaration that leaked either dropped key would be text the model
    /// never saw in training.
    #[test]
    fn the_tools_turn_unwraps_the_envelope_and_drops_the_two_keys() {
        let tools = json!([{
            "type": "function",
            "function": { "name": "read", "strict": true, "defer_loading": false }
        }]);
        let turn = tools_system_turn(&tools);
        assert!(
            turn.contains("\n<tools>\n{\"name\": \"read\"}\n</tools>\n"),
            "{turn}"
        );
        assert!(
            !turn.contains("strict") && !turn.contains("defer_loading"),
            "{turn}"
        );
    }

    /// A missing `eos_token_id` is `Ok(vec![])` and an unreadable file is an `Err` — the
    /// 2026-08-13 split this function's doc argues for, which only a test can hold apart
    /// (both spellings reach `Tokenizer::load` as "no stop tokens" and diverge only here).
    #[test]
    fn a_missing_eos_key_is_empty_and_an_absent_file_is_an_error() {
        // Salted by pid, and created FRESH: a fixed path poisons every later run if an
        // assertion panics after the first write, and this machine runs several test
        // trees concurrently against one /tmp (review 2026-08-16).
        let dir = std::env::temp_dir().join(format!("rivoli-tok-eos-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let d = dir.to_str().unwrap();
        assert!(eos_token_ids(d).is_err(), "no generation_config.json yet");

        let p = dir.join("generation_config.json");
        std::fs::write(&p, br#"{"pad_token_id": 1}"#).unwrap();
        assert_eq!(eos_token_ids(d).unwrap(), Vec::<u32>::new());

        // A bare int and an array are both the artifact's own spelling of the same fact.
        std::fs::write(&p, br#"{"eos_token_id": 7}"#).unwrap();
        assert_eq!(eos_token_ids(d).unwrap(), vec![7]);
        std::fs::write(&p, br#"{"eos_token_id": [7, 9]}"#).unwrap();
        assert_eq!(eos_token_ids(d).unwrap(), vec![7, 9]);

        std::fs::write(&p, b"not json").unwrap();
        assert!(eos_token_ids(d).is_err(), "unreadable content is an Err");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
