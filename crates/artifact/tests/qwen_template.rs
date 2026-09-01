//! The chat-template pin: [`rivoli_artifact::qwen_encoding::render`] against
//! Qwen3.8-Flash-Next's own `chat_template.jinja`, rendered by the model's own tokenizer.
//!
//! **The expected side is not a reading of the template — it is the template's output.**
//! `qwen_template_driver.py`, vendored beside this file, runs
//! `AutoTokenizer.apply_chat_template` at revision `236dfdf2` over 78 cases and writes
//! `(kwargs, expected, ids)` — or `(kwargs, raises)` — into `qwen-chat-cases.json`. That is
//! what makes this a pin rather than a second transcription: GLM's hand-port drifted to GLM-4's
//! framing and survived months of review because nothing ever compared it against the
//! checkpoint's own file (`tokenizer.rs`'s dated correction, and the
//! `artifact-drops-the-chat-template` note).
//!
//! **78 cases, not the plan's ~31**, split **65 rendering + 13 refusals** — the only three
//! counts here, and each is asserted rather than stated ([`cases`], and the `scored` tallies in
//! the two loops below). `docs/investigations/qwen-flash-next-port.md`'s exit-gate row was
//! written before the template was read; three surfaces it does not name account for the
//! difference, and none of them collapses into fewer cases:
//!
//! - **`reasoning_effort`** — THREE legal values, `xhigh` is the DEFAULT and `medium` is the
//!   only one that emits nothing, so a render with no kwargs at all already carries a
//!   synthesised system turn; anything else refuses; and the whole block is inert when thinking
//!   is off, so a bogus value there does NOT refuse.
//! - **`enable_thinking` and `preserve_thinking`** — FOUR states each, because Jinja's
//!   `is true`/`is false` are IDENTITY against the booleans, so `1`, `0`, `null` and `"true"`
//!   land in a state neither branch was written for.
//! - **the reject direction** — NINE `raise_exception` sites, three of them reachable from an
//!   ordinary OpenAI client. The count is a gate here, not a sentence: five files in this port
//!   first said eight (review 2026-09-01), which is the prose-number class this repo punishes.
//!
//! The rest are the surfaces the row did name — framing, system turns, multi-turn, tools, tool
//! calls, tool results, content parts, the trailing generation prompt — plus the vision
//! placeholders and the `|trim` rule. A per-surface breakdown is deliberately NOT tabulated
//! here: cases carry several kwargs at once, so any such table is an attribution choice nothing
//! recomputes, and this file already had two mutually inconsistent versions of one.
//!
//! **The fixture and the driver are vendored together.** Vendoring the bytes without the
//! program that produced them makes a pin nobody can re-derive; the anchor fixtures in
//! `crates/oracles` carry their regeneration script for the same reason. Regenerating needs
//! 12,809,320 B of tokenizer and a pinned transformers install and is therefore not something this
//! suite does.
//!
//! No GPU, no lock, no network, no Python — the fixture and the template are bytes in the tree.
#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

use std::{collections::BTreeSet, io::Write};

use rivoli_artifact::qwen_encoding::{
    IM_END, IM_START, JinjaFlag, QwenChatOpts, THINK_CLOSE, THINK_OPEN, render,
};
use serde_json::Value;

/// The reference's own output, `(kwargs, expected, ids)` or `(kwargs, raises)` per case.
const CASES: &str = include_str!("qwen-chat-cases.json");

/// **The template itself, in tree.** So the port's long literals are scored against the
/// checkpoint's file and not only against a fixture generated from it: the fixture and the
/// template could drift together if the fixture were regenerated from a different revision,
/// and [`the_vendored_template_is_the_pinned_revisions_own_file`] is what stops that.
const TEMPLATE: &str = include_str!("../../../docs/measurement/qwen-reference/chat_template.jinja");

/// Track A's census, read (never written) so the two pins of the SAME file must agree.
const CENSUS: &str = include_str!("../../../docs/measurement/qwen-reference/tensor-families.tsv");

/// The whole fixture document — provenance and cases together, because two tests need the
/// provenance and every test needs the cases.
fn doc() -> Value {
    serde_json::from_str(CASES).expect("qwen-chat-cases.json parses")
}

/// Every case, with the anti-vacuity count applied at the ONE place that reads them.
///
/// **Hoisted here rather than repeated per test** — a fixture emptied by a failed driver run, or
/// an `include_str!` of a truncated file, would otherwise leave every loop below green having
/// compared nothing. Asserted on the INPUT rather than on a counter each loop increments: no
/// loop here has a `continue`, so the two are provably equal, and the counter form is what
/// `glimmer_template.rs`'s own review found to be a verbatim copy escaping the duplication gate
/// only by sitting under its 15-token floor.
fn cases(d: &Value) -> &Vec<Value> {
    let all = d["cases"].as_array().expect("cases is an array");
    assert_eq!(all.len(), 78, "expected 78 vendored cases");
    all
}

/// A case's name, for the panic messages.
fn name_of(case: &Value) -> &str {
    case["name"].as_str().expect("case has a name")
}

/// Rebuild [`QwenChatOpts`] from a case's recorded kwargs, so the Rust side is driven by
/// exactly the arguments the Python side was.
///
/// **The three-way `get` is the whole point of [`JinjaFlag`].** An absent key, a JSON `null`
/// and a JSON `1` are three different template states, and `and_then(Value::as_bool)` maps the
/// last two onto the same `None` — which would silently move a case into the branch it exists
/// to distinguish.
fn opts_of(kw: &Value) -> QwenChatOpts<'_> {
    QwenChatOpts {
        add_generation_prompt: flag(kw, "add_generation_prompt"),
        enable_thinking: JinjaFlag::of(kw.get("enable_thinking")),
        preserve_thinking: JinjaFlag::of(kw.get("preserve_thinking")),
        reasoning_effort: kw.get("reasoning_effort"),
        tools: kw.get("tools"),
        add_vision_id: flag(kw, "add_vision_id"),
    }
}

/// A boolean kwarg, absent meaning `false` — `apply_chat_template`'s own default for both of
/// the two this fixture varies.
///
/// **Spelled once because it is spelled three times in `glimmer_template.rs`**, and jscpd
/// matched this chain across the two files at 34, 39 and 168 tokens on the first compile. The
/// template applies PYTHON truthiness to `add_generation_prompt`, so `1` would be true there and
/// `false` here; no vendored case sets either flag to a non-bool, and the port takes a `bool`
/// precisely so the caller owns that decision.
fn flag(kw: &Value, key: &str) -> bool {
    kw.get(key).and_then(Value::as_bool).unwrap_or(false)
}

/// The messages a case renders.
fn messages_of(kw: &Value) -> &Vec<Value> {
    kw["messages"].as_array().expect("kwargs has messages")
}

/// A case's vendored ids.
///
/// Deserialized straight into `Vec<u32>` rather than walked element by element: the walk is what
/// `glimmer_template.rs` spells out, and a second copy of that `as_array`/`as_u64`/`as u32`
/// chain is a clone jscpd already reported once between two tests in the SAME file. Serde does
/// the range check for free and says so on failure.
fn ids_of(case: &Value) -> Vec<u32> {
    serde_json::from_value(case["ids"].clone()).expect("case ids are u32s")
}

/// The index of the first element where two sequences differ, or where one ends.
///
/// Three lines and shared by the byte pin and the id pin, because the alternative — one reporter
/// per element type — is the pair jscpd matched at 65 tokens in `glimmer_template.rs`. The
/// FORMATTING is per test rather than a `show` closure: a byte window wants escaped text and an
/// id window wants the numbers, and that difference is the whole reason a closure existed there.
fn first_diff<T: PartialEq>(got: &[T], want: &[T]) -> Option<usize> {
    let common = got.iter().zip(want).position(|(a, b)| a != b);
    common.or_else(|| (got.len() != want.len()).then(|| got.len().min(want.len())))
}

/// The point of the whole file: every rendering case, byte for byte.
///
/// Reported as a first-difference offset with both sides' context rather than as two whole
/// strings — the tools cases are 1.4 KB of identical protocol preamble, and a bare `assert_eq!`
/// on those is a diff nobody can read.
#[test]
fn every_rendering_case_renders_byte_for_byte() {
    let d = doc();
    let mut scored = 0usize;
    for case in cases(&d) {
        let Some(want) = case.get("expected").and_then(Value::as_str) else {
            continue; // a refusal case; `every_refusal_carries_the_templates_own_message` owns it
        };
        let (name, kw) = (name_of(case), &case["kwargs"]);
        let got = render(messages_of(kw), &opts_of(kw))
            .unwrap_or_else(|e| panic!("case `{name}` must render, but refused: {e}"));
        if let Some(at) = first_diff(got.as_bytes(), want.as_bytes()) {
            let lo = at.saturating_sub(60);
            panic!(
                "case `{name}` diverges at byte {at} (got {} bytes, want {})\n  got  \
                 ...{:?}\n  want ...{:?}",
                got.len(),
                want.len(),
                String::from_utf8_lossy(&got.as_bytes()[lo..(at + 60).min(got.len())]),
                String::from_utf8_lossy(&want.as_bytes()[lo..(at + 60).min(want.len())]),
            );
        }
        scored += 1;
    }
    // The rendering cases and the refusal cases must partition the fixture, so neither test can
    // silently score zero while the other looks busy.
    assert_eq!(scored, 65, "expected 65 rendering cases");
}

/// **The reject direction, which is the one a strict port gets wrong.** Every case the reference
/// refused must be refused here with the SAME message.
///
/// This template calls `raise_exception` in nine places and three are reachable from an
/// ordinary OpenAI client: a `developer` role is `Unexpected message role.` (Glimmer's template
/// silently DROPS an unknown role, so a port that copied its neighbour would invent a framing
/// the model has never seen), a conversation with no real user turn is `No user query found in
/// messages.`, and a system turn that is not first is `System message must be at the beginning.`
///
/// `refuse_content_before_role` is here for the ORDERING rather than for the message: the
/// template renders content at the top of the loop body and checks the role after it, so a
/// malformed content part on a bogus role reports the CONTENT error. A port that validated the
/// role first would refuse both cases and pass a test that only checked "it refused".
/// Which of the template's nine refusal sites the fixture actually reaches.
///
/// **The count gate in [`the_vendored_template_is_the_pinned_revisions_own_file`] says there are
/// nine sites; this says eight of them are covered and names the ninth.** Each message is
/// checked against the TEMPLATE's own bytes as well as against the cases, so the list is
/// anchored at both ends rather than being a third transcription: a site the template no longer
/// raises reddens here, and so does a regenerated fixture that dropped a refusal.
///
/// `No messages provided.` is the site no vendored case can carry, and it is unreachable by
/// CONSTRUCTION rather than by omission: `apply_chat_template` raises `ValueError: Cannot apply
/// chat template to an empty conversation.` before the template is entered (measured
/// 2026-08-31), so the reference has no answer to record for it. `render` implements the
/// template's own text anyway, and the last assertion here is the ONE self-asserted refusal in
/// this file — labelled, because "the reference agrees" is the one claim it cannot make.
fn check_every_refusal_site_is_covered(all: &[Value]) {
    // `Unexpected reasoning effort ` ends at a space because the template interpolates the
    // offending value with `~` there; that prefix is the whole of the site's own literal.
    const COVERED: [&str; 8] = [
        "System message cannot contain images.",
        "System message cannot contain videos.",
        "Unexpected item type in content.",
        "Unexpected content type.",
        "Unexpected reasoning effort ",
        "No user query found in messages.",
        "System message must be at the beginning.",
        "Unexpected message role.",
    ];
    const UNREACHABLE: &str = "No messages provided.";
    let raised = |site: &str| {
        all.iter()
            .filter_map(|c| c.get("raises").and_then(Value::as_str))
            .any(|r| r.starts_with(site))
    };
    for site in COVERED {
        assert!(
            TEMPLATE.contains(site),
            "the vendored template no longer raises {site:?}"
        );
        assert!(
            raised(site),
            "no vendored case covers the refusal site {site:?}"
        );
    }
    assert!(TEMPLATE.contains(UNREACHABLE));
    assert!(
        !raised(UNREACHABLE),
        "a vendored case now claims {UNREACHABLE:?} — the reference refuses an empty \
         conversation BEFORE entering the template, so it cannot have produced that text"
    );
    // `Value::Null` is "no kwargs at all": every `get` on it is `None`, so this is the default
    // render, and the only thing varied is that there are no messages.
    let refused = render(&[], &opts_of(&Value::Null)).expect_err("an empty conversation refuses");
    assert_eq!(refused.to_string(), UNREACHABLE, "the self-asserted site");
}

#[test]
fn every_refusal_carries_the_templates_own_message() {
    let d = doc();
    // Ahead of the per-case loop: both reach a fixture that lost a refusal, and the census names
    // the SITE that lost its coverage where the loop names only the case whose text moved.
    check_every_refusal_site_is_covered(cases(&d));
    let mut scored = 0usize;
    for case in cases(&d) {
        let Some(want) = case.get("raises").and_then(Value::as_str) else {
            continue;
        };
        let (name, kw) = (name_of(case), &case["kwargs"]);
        assert_eq!(
            case["raises_type"], "TemplateError",
            "case `{name}`: only the template's own raise_exception is a contract"
        );
        match render(messages_of(kw), &opts_of(kw)) {
            Ok(text) => panic!("case `{name}` must refuse with {want:?}, rendered {text:?}"),
            Err(e) => assert_eq!(e.to_string(), want, "case `{name}` refusal text"),
        }
        scored += 1;
    }
    assert_eq!(scored, 13, "expected 13 refusal cases");
}

/// **The id pin, and it is the gate the whole framing rests on**: `render`'s bytes through the
/// SHIPPED tokenizer must equal the ids `apply_chat_template` produced.
///
/// **Why it matters more than the byte pin.** `render` returns a STRING containing
/// `<|im_start|>`, `<think>`, `<tool_call>` and friends as literal text, and whether each becomes
/// ONE id is decided by the `tokenizers` crate reading the checkpoint's added-token table — not
/// by anything in this crate. The byte pin cannot see that; a port whose specials tokenized as
/// five ordinary pieces each would pass it and feed the model a prompt it has never seen. It is
/// not a formality here: `<think>`, `</think>`, `<tool_call>` and `<tool_response>` are added
/// tokens with **`special: false`** (measured from `tokenizer_config.json`'s
/// `added_tokens_decoder`, 2026-08-31), a flag combination none of the four shipped
/// architectures uses for a token the template emits.
///
/// **The tokenizer is 12,809,320 B and is not vendored**, so the directory is supplied by
/// `RIVOLI_QWEN_ARTIFACT` (a Qwen artifact, or the HF checkpoint — they carry identical
/// `tokenizer.json`, sha256 `0997f410…`, which is the file the fixture's provenance pins and
/// the one this pin was last run against). Without it the id half cannot run, and
/// [`skipped_id_pin`] is what stops that from reading as a pass.
///
/// **Nothing here claims CI arms this, because CI does not.** An earlier version of this
/// comment said `RIVOLI_QWEN_REQUIRED=1` "is what CI and the closeout run set":
/// `.github/workflows/ci.yml` sets exactly one such variable, `RIVOLI_CS_REQUIRED` (line 85),
/// this one appears in no workflow, and a CI runner has no 12.8 MB tokenizer to point
/// `RIVOLI_QWEN_ARTIFACT` at — so the sentence told the next reader the id pin was armed
/// somewhere it has never run, while a clean run had it green having tokenized NOTHING (review
/// 2026-09-01). What arms it is a closeout run that sets both, which is all the
/// `drafter_convert.rs` precedent claims for its own.
#[test]
fn rendered_prompts_tokenize_to_the_vendored_ids() {
    let d = doc();
    let all = cases(&d); // covers BOTH branches below, so an unset variable cannot read as coverage
    // Counted BEFORE the branch: the skip path can then say how big the hole is, and neither
    // path can be green over a fixture that quietly stopped carrying ids.
    let id_bearing = all.iter().filter(|c| c.get("ids").is_some()).count();
    assert_eq!(id_bearing, 65, "expected 65 id-bearing cases");
    let dir = match std::env::var("RIVOLI_QWEN_ARTIFACT") {
        Ok(dir) => dir,
        Err(_) => {
            skipped_id_pin(id_bearing);
            return;
        }
    };
    let tok = rivoli_artifact::tokenizer::Tokenizer::load(&dir)
        .unwrap_or_else(|e| panic!("RIVOLI_QWEN_ARTIFACT={dir} is not loadable: {e}"));
    let mut scored = 0usize;
    for case in all {
        if case.get("ids").is_none() {
            continue;
        }
        let (name, kw) = (name_of(case), &case["kwargs"]);
        let want = ids_of(case);
        let text = render(messages_of(kw), &opts_of(kw)).expect("a case with ids renders");
        let got = tok
            .encode(&text)
            .unwrap_or_else(|e| panic!("case `{name}`: encode failed: {e}"));
        if let Some(at) = first_diff(&got, &want) {
            let lo = at.saturating_sub(6);
            panic!(
                "case `{name}` diverges at id {at} (got {} ids, want {})\n  got  ...{:?}\n  \
                 want ...{:?}",
                got.len(),
                want.len(),
                &got[lo..(at + 6).min(got.len())],
                &want[lo..(at + 6).min(want.len())],
            );
        }
        scored += 1;
    }
    // Against the count taken before the branch rather than against a second literal 65. The
    // anti-vacuity check is the one ABOVE, which both paths run; this one exists because the
    // predicate is written twice — `is_some` up there, `is_none()` + `continue` here — and it is
    // what stops the two from drifting apart.
    assert_eq!(scored, id_bearing, "an id-bearing case was not examined");
    println!("  id pin: {scored} cases tokenized identically to apply_chat_template");
}

/// The id pin's skip path: no tokenizer on this machine, so state the size of the hole and
/// refuse to be quiet about it when the caller said the pin was required.
///
/// **`RIVOLI_QWEN_REQUIRED` is read by VALUE, not by existence.** `crates/cli/tests/codescene.rs`
/// became `is_some_and(|v| !v.is_empty())` in commit `051a291` for exactly this: a GitHub Actions
/// `env:` expression that evaluates to `''` still SETS the variable, and run 33383324860 armed a
/// REQUIRED mode nobody had configured. This is the same contract, two commits later.
///
/// **The notice is written to the `Stderr` HANDLE, not through `eprintln!`.** libtest's output
/// capture is consulted by the print macros and not by a direct handle write, so this line is
/// visible in a plain `cargo test` run where the macro's is not. Measured 2026-09-01 on rustc
/// 1.96.0, one passing test emitting both forms: without `--nocapture` the macro line appears 0
/// times and the handle line once; with it, both. That is the half the `libtest-captures-SKIP-
/// lines` note says a printed skip cannot do — and the asserted count above is the half that
/// does not depend on anyone reading the run at all.
fn skipped_id_pin(not_examined: usize) {
    let required = std::env::var_os("RIVOLI_QWEN_REQUIRED").is_some_and(|v| !v.is_empty());
    assert!(
        !required,
        "qwen id pin REQUIRED but did not run: RIVOLI_QWEN_ARTIFACT is unset, so {not_examined} \
         id-bearing cases were NOT examined"
    );
    let _ = writeln!(
        std::io::stderr(),
        "SKIP qwen id pin: RIVOLI_QWEN_ARTIFACT unset — {not_examined} id-bearing cases NOT \
         examined (set RIVOLI_QWEN_REQUIRED=1 to make this a failure)"
    );
}

/// **The one deliberate divergence, pinned against the reference rather than asserted.**
///
/// A `tool_call.arguments` that is a JSON STRING — what OpenAI, Azure and every SDK mirroring
/// them put on the wire — makes the reference raise: `arguments|items` is Jinja's `items`
/// filter, which is `TypeError: Can only get item pairs from a mapping.` on a `str` (measured
/// 2026-08-31 against the pinned revision). So there are no reference bytes for the string form
/// and the divergence could only ever be self-asserted — except through the OBJECT form, which
/// the reference renders happily. This renders the string form and requires it to equal the
/// **object** case's vendored reference bytes, which is the strongest statement available:
/// the port turns the wire format into exactly the prompt the reference produces for the same
/// call.
///
/// The alternative to parsing was refusing every replayed tool conversation. The third option —
/// drop the arguments and emit a syntactically valid call to the right function with nothing in
/// it — is measured in `glimmer_encoding::atem`'s own note as teaching the model that calling a
/// tool with no arguments is normal.
#[test]
fn a_json_string_arguments_renders_as_the_object_the_reference_was_given() {
    let d = doc();
    let case = cases(&d)
        .iter()
        .find(|c| name_of(c) == "tool_call_args_object_for_the_string_divergence")
        .expect("the object-form case is in the fixture");
    let want = case["expected"].as_str().expect("it renders");
    let object_form = &case["kwargs"]["messages"];
    // The SAME conversation with `arguments` respelled as the wire format's JSON string. Built
    // from the fixture's own messages so the two cannot drift apart: only `arguments` changes.
    let mut string_form = object_form.clone();
    let args = string_form[1]["tool_calls"][0]["function"]["arguments"].take();
    let json = serde_json::to_string(&args).expect("re-serialize the arguments");
    string_form[1]["tool_calls"][0]["function"]["arguments"] = Value::String(json);
    assert_ne!(
        &string_form, object_form,
        "the respelling must change something"
    );
    let got = render(
        string_form.as_array().expect("messages"),
        &opts_of(&case["kwargs"]),
    )
    .expect("the string form renders");
    assert_eq!(
        got, want,
        "the JSON-string form must render as the object form"
    );
}

/// **The vendored template IS the pinned revision's file, recomputed from the bytes on disk.**
///
/// A pin checked only against a frozen copy of itself is decoration, so this recomputes FNV-1a
/// over `docs/measurement/qwen-reference/chat_template.jinja` and compares it to the hash the
/// FIXTURE records — the fixture being the only other place that value exists — and then to
/// track A's census, which pinned the same file before it was vendored. Three independent
/// homes for one number, and none of them can move alone.
///
/// FNV-1a rather than the sha256 the fixture also carries: `rivoli_core::hash::fnv1a` is in the
/// tree and `sha2` is not, and adding a crypto dependency to pin a fixture is the worse trade
/// (`k3_tokenizer.rs` states the same rule). The sha256 stays in the JSON and the tsv for a
/// human comparing against upstream, and the byte count is checked here because a hash agrees
/// on nothing a length disagrees about.
///
/// **The literal check is the half a hash cannot do.** A hash says the file is unchanged; it
/// says nothing about whether the port's long constants came out of it. So each is searched for
/// in the template's own bytes. `TOOLS_FORMAT` and the two effort sentences are 200–1000
/// character literals where a single typo shifts a system prompt, and only the cases that
/// happen to exercise them would otherwise notice.
#[test]
fn the_vendored_template_is_the_pinned_revisions_own_file() {
    let d = doc();
    let p = &d["provenance"];
    // **The refusal-site count, gated rather than stated.** Six files in this port said "eight"
    // and nothing recomputed it (review 2026-09-01); the template raises at NINE sites, and a
    // revision that adds a tenth is a refusal this port does not implement and no case covers.
    // Deliberately AHEAD of the length and hash asserts: every plant that changes this count
    // also changes the bytes, so behind them this line could never be observed red.
    assert_eq!(
        TEMPLATE.matches("raise_exception").count(),
        9,
        "the vendored template's raise_exception sites"
    );
    assert_eq!(TEMPLATE.len(), 8952, "the vendored template's byte count");
    assert_eq!(
        p["chat_template.jinja"]["bytes"].as_u64(),
        Some(TEMPLATE.len() as u64)
    );
    let recomputed = format!("{:016x}", rivoli_core::hash::fnv1a(TEMPLATE.as_bytes()));
    // Recomputed on the LEFT, pinned on the right, so a red reads as got-vs-want.
    assert_eq!(
        Some(recomputed.as_str()),
        p["chat_template.jinja"]["fnv1a64"].as_str(),
        "the vendored template's recomputed hash is not the one the fixture pins"
    );
    // Track A pinned this file's sha256 and fnv1a64 in the census BEFORE it was vendored. Both
    // must still be the same file; a regeneration against another revision reddens here.
    for pin in ["sha256", "fnv1a64"] {
        let v = p["chat_template.jinja"][pin]
            .as_str()
            .expect("the fixture records both hashes");
        assert!(
            CENSUS.contains(v),
            "tensor-families.tsv does not carry the template's {pin} {v}"
        );
    }
    let rev = p["revision"].as_str().expect("a revision");
    assert!(CENSUS.contains(rev), "the census pins another revision");
    // The port's literals, in the template's own bytes. Resolved through the public consts and
    // through `render` rather than restated, so this cannot pass against a typo in either.
    for literal in [IM_START, IM_END, THINK_OPEN, THINK_CLOSE] {
        assert!(TEMPLATE.contains(literal), "template lacks {literal}");
    }
    for fragment in [
        "Reasoning effort is set to xhigh.",
        "Reasoning effort is set to low.",
        "# Tools\\n\\nYou have access to the following functions:\\n\\n<tools>",
        "an inner <function=...></function> block must be nested within <tool_call></tool_call>",
        "Supported types are xhigh (default), medium, and low.",
    ] {
        assert!(TEMPLATE.contains(fragment), "template lacks {fragment:?}");
    }
}

/// Every special the template can emit is exercised by some case, and every rendered prompt
/// starts at `<|im_start|>` — a CENSUS of the fixture.
///
/// **There is no BOS**, which is the property the first assertion really pins: `bos_token` is
/// `null` and `add_bos_token` is `false`, and four of the five architectures here DO have one.
/// A port that prepended one would shift every id in every case, and would look like a
/// tokenizer problem.
#[test]
fn every_special_the_template_emits_is_exercised_by_some_case() {
    // `<|im_start|>`, `<|im_end|>`, `<think>`, `</think>`, `<tool_call>`, `</tool_call>`,
    // `<tool_response>`, `</tool_response>`, `<|vision_start|>`, `<|vision_end|>`,
    // `<|image_pad|>`, `<|video_pad|>` — resolved from the vendored ids rather than restated,
    // since the whole question is what the tokenizer does with them.
    const SPECIALS: [u32; 12] = [
        248_045, 248_046, 248_068, 248_069, 248_058, 248_059, 248_066, 248_067, 248_053, 248_054,
        248_056, 248_057,
    ];
    let d = doc();
    let mut seen: BTreeSet<u32> = BTreeSet::new();
    for case in cases(&d) {
        if case.get("ids").is_none() {
            continue;
        }
        let ids = ids_of(case);
        assert_eq!(
            ids.first(),
            Some(&248_045),
            "case `{}` must start at <|im_start|> — this checkpoint has NO bos",
            name_of(case)
        );
        seen.extend(ids);
    }
    // EVERY id goes into the set, not just the specials: the missing-set form below names all of
    // them at once instead of the first, and the same ordered set then carries the vocabulary
    // bound for free. Filtering on the way in was `glimmer_template.rs`'s line verbatim, which
    // jscpd matched at 34 tokens.
    let missing: Vec<u32> = SPECIALS
        .iter()
        .copied()
        .filter(|s| !seen.contains(s))
        .collect();
    assert!(
        missing.is_empty(),
        "no vendored case exercises special ids {missing:?}"
    );
    // Every id the fixture contains is inside the vocabulary. `248_320` is `vocab_size` from the
    // vendored `config.json`'s `text_config`; the highest added token is 248,076, so a fixture
    // that had picked up ids from another checkpoint's vocabulary would be caught here rather
    // than at the first decode.
    let &max_id = seen.last().expect("the fixture carries ids");
    assert!(max_id < 248_320, "id {max_id} is outside vocab_size 248320");
}

/// The three axes each take enough distinct values that a port ignoring one would be caught.
///
/// **Presence is not variation** (`glimmer_template.rs`'s recorded review, 2026-08-14): a case
/// that set `reasoning_effort: "xhigh"` — the template's own default — satisfies "some case sets
/// it" while leaving `render` free to ignore the option entirely. So the values are checked
/// against the defaults, and every state of the two flags is required by NAME.
fn check_every_flag_state_appears(all: &[Value]) {
    for field in ["enable_thinking", "preserve_thinking"] {
        let states: BTreeSet<JinjaFlag> = all
            .iter()
            .map(|c| JinjaFlag::of(c["kwargs"].get(field)))
            .collect();
        for want in [
            JinjaFlag::Undefined,
            JinjaFlag::True,
            JinjaFlag::False,
            JinjaFlag::OtherDefined,
        ] {
            assert!(
                states.contains(&want),
                "no case has {field} in state {want:?} — Jinja's `is true` is IDENTITY, so the \
                 fourth state is a real branch and an untested one"
            );
        }
    }
}

/// All three legal reasoning efforts, plus the absent kwarg, plus a refusal.
///
/// `medium` is the one that must be present for the right reason: it is the only legal value
/// that emits NO instruction, so without it "the effort text is a constant" is untested.
fn check_every_effort_level_appears(all: &[Value]) {
    let efforts: BTreeSet<Option<&str>> = all
        .iter()
        .map(|c| c["kwargs"].get("reasoning_effort").and_then(Value::as_str))
        .collect();
    for want in [None, Some("xhigh"), Some("medium"), Some("low")] {
        assert!(
            efforts.contains(&want),
            "no case sets reasoning_effort {want:?}"
        );
    }
    assert!(
        all.iter().any(|c| {
            c.get("raises")
                .and_then(Value::as_str)
                .is_some_and(|m| m.starts_with("Unexpected reasoning effort"))
        }),
        "no case refuses an illegal reasoning_effort"
    );
}

/// All four `tools` shapes, and both values of the two bools.
///
/// `tools` has no scalar default to differ from; what matters is that BOTH truthiness arms are
/// present, since an empty array and `null` are FALSE in Python and were the branch a review
/// found emitting 1277 bytes the reference omits — on a preamble smaller than this one.
fn check_every_tools_shape_and_both_bools(all: &[Value]) {
    // Four independent predicates rather than a set of shape NAMES: the name-collecting form was
    // byte-identical to `glimmer_template.rs`'s and jscpd refused the build, and asking each
    // question separately means a failure names the missing shape instead of the set.
    let some_case =
        |pred: fn(Option<&Value>) -> bool| all.iter().any(|c| pred(c["kwargs"].get("tools")));
    assert!(some_case(|t| t.is_none()), "no case omits `tools`");
    assert!(
        some_case(|t| t == Some(&Value::Null)),
        "no case sets `tools` to null — Python truthiness and Rust Option-ness disagree on it"
    );
    assert!(
        some_case(|t| t.is_some_and(|v| v.as_array().is_some_and(Vec::is_empty))),
        "no case sets `tools` to [] — the shape that emitted 1277 bytes of preamble on Glimmer"
    );
    assert!(
        some_case(|t| t.is_some_and(|v| v.as_array().is_some_and(|a| !a.is_empty()))),
        "no case sets a populated `tools`"
    );
    for field in ["add_generation_prompt", "add_vision_id"] {
        let values: BTreeSet<bool> = all.iter().map(|c| flag(&c["kwargs"], field)).collect();
        assert_eq!(values.len(), 2, "every case sets the same {field}");
    }
}

/// The pin's own anti-vacuity: the recorded kwargs must actually reach the render.
///
/// A `render` that ignored `opts` entirely would fail the byte comparison, but a `render` that
/// ignored ONE option would only fail if some case varies it. This asserts the variation exists,
/// which is the property the byte test silently depends on.
///
/// **A driver over named `check_*` helpers rather than one long body.** That is a code-health
/// requirement rather than taste — the CodeScene gate refuses a long unbroken run of assertions,
/// and a run of asserts with no name on it makes a failure read as "something in this file
/// broke".
#[test]
fn the_fixture_varies_every_option() {
    let d = doc();
    let all = cases(&d);
    check_every_flag_state_appears(all);
    check_every_effort_level_appears(all);
    check_every_tools_shape_and_both_bools(all);
}

/// The fixture was generated by the pinned toolchain, recorded so a regeneration under a
/// different Jinja cannot pass silently.
///
/// **Not a formality: the renderer is half the reference.** `tojson` is transformers' own
/// override (`json.dumps(..., ensure_ascii=False)`, not Jinja's HTML-safe one), `trim_blocks`
/// and `lstrip_blocks` are set by transformers, and `is true`/`is false` became identity tests
/// in Jinja 2.11 — three behaviours the template's bytes do not carry and this port depends on.
/// A fixture regenerated under a different stack is a different oracle.
#[test]
fn the_fixture_records_the_stack_that_produced_it() {
    let d = doc();
    let p = &d["provenance"];
    for (key, want) in [
        ("repo", "Qwen/Qwen3.8-Flash-Next-FP8"),
        ("revision", "236dfdf285828023ca3bcd3f37366c58a3469b13"),
        ("transformers", "5.16.1"),
        ("jinja2", "3.1.6"),
    ] {
        assert_eq!(p[key].as_str(), Some(want), "provenance {key}");
    }
    // The two homes of this template in the source repo are EXACTLY byte-identical — not
    // identical-modulo-a-trailing-newline, which is what the plan of record's census item 2
    // says. The .jinja file carries no trailing newline, so `tokenizer_config.json`'s
    // `chat_template` key is the same 8952 characters. Measured 2026-08-31; asserted here
    // because the vendored copy is the .jinja one and a converter may copy either.
    assert!(
        !TEMPLATE.ends_with('\n'),
        "the vendored template must be the .jinja bytes, whose last line is the outer endif"
    );
    assert!(TEMPLATE.ends_with("{%- endif %}"));
}
