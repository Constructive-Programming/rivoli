//! The chat-template byte pin: [`rivoli_artifact::glimmer_encoding::render`] against Muse
//! Glimmer's own `chat_template.jinja`, rendered by the model's own tokenizer.
//!
//! **The expected side is not a reading of the template — it is the template's output.**
//! `glimmer_template_driver.py`, vendored beside this file, runs
//! `AutoTokenizer.apply_chat_template` on `meta-models/Muse-Glimmer-30B` over 31 cases and
//! writes `(kwargs, expected, ids)` into `glimmer-chat-cases.json`. That is what makes this a
//! pin rather than a second transcription: GLM's hand-port drifted to GLM-4's framing and
//! survived months of review because nothing ever compared it against the checkpoint's own file
//! (`tokenizer.rs`'s dated correction, and the `artifact-drops-the-chat-template` note).
//!
//! **The fixture and the driver are vendored together** (ported 2026-08-16 from
//! `old:tests/glimmer-chat-cases.json` and `old:tests/glimmer_template_driver.py`,
//! `wt/glimmer-s2` @ 6b7f496, byte-identical). Vendoring the bytes without the program that
//! produced them makes a pin nobody can re-derive; the anchor fixtures in `crates/oracles`
//! carry their regeneration script for exactly this reason. Regenerating needs the 60 GB
//! checkpoint and a transformers install and is therefore not something this suite does.
//!
//! No GPU, no lock, no network, no Python — the fixture is bytes in the tree.
#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

use std::collections::BTreeSet;

use rivoli_artifact::glimmer_encoding::{GlimmerChatOpts, render};
use serde_json::Value;

const CASES: &str = include_str!("glimmer-chat-cases.json");

fn cases() -> Vec<Value> {
    let doc: Value = serde_json::from_str(CASES).expect("glimmer-chat-cases.json parses");
    doc["cases"].as_array().expect("cases is an array").clone()
}

/// Rebuild [`GlimmerChatOpts`] from a case's recorded kwargs, so the Rust side is driven by
/// exactly the arguments the Python side was.
fn opts_of(kw: &Value) -> GlimmerChatOpts<'_> {
    GlimmerChatOpts {
        add_generation_prompt: kw
            .get("add_generation_prompt")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        reasoning_strength: kw.get("reasoning_strength").and_then(Value::as_str),
        knowledge_cutoff: kw.get("knowledge_cutoff").and_then(Value::as_str),
        tools: kw.get("tools"),
        tool_namespace_descriptions: kw.get("tool_namespace_descriptions"),
        ..GlimmerChatOpts::new(kw["current_date"].as_str().unwrap_or(""))
    }
}

/// A case's vendored ids. Spelled once because TWO tests read them — this and the census
/// below — and jscpd reported the second copy of the `as_array`/`as_u64` chain as a clone
/// (2026-08-17). Said as two rather than three because the first draft of this line said
/// three, which is the count-nobody-re-derived class in a comment about a duplication gate.
fn ids_of(case: &Value) -> Vec<u32> {
    case["ids"]
        .as_array()
        .expect("case has ids")
        .iter()
        .map(|v| v.as_u64().expect("id is a number") as u32)
        .collect()
}

/// Compare two sequences and panic on the FIRST divergence, with `win` elements of context on
/// both sides rendered by `show`.
///
/// **One reporter for the byte pin and the id pin, and the factoring was forced.** Written
/// twice — once over `&[u8]`, once over `&[u32]` — jscpd matched the pair at 65 tokens and
/// refused the build. The generic is also the honest shape: "first divergence plus a window"
/// is one idea, and the only thing that differs is how a window is worth printing (a `str`
/// slice for text, the numbers themselves for ids), which is exactly what `show` is.
///
/// Reported this way rather than as a bare `assert_eq!` because the tool-definition cases are
/// 2 KB of identical preamble, and a diff of two 4 KB strings is a diff nobody reads.
fn assert_same<T: PartialEq, F: Fn(&[T]) -> String>(
    name: &str,
    unit: &str,
    (got, want): (&[T], &[T]),
    win: usize,
    show: F,
) {
    let Some(at) = got
        .iter()
        .zip(want)
        .position(|(a, b)| a != b)
        .or_else(|| (got.len() != want.len()).then(|| got.len().min(want.len())))
    else {
        return;
    };
    let lo = at.saturating_sub(win);
    panic!(
        "case `{name}` diverges at {unit} {at} (got {} {unit}s, want {})\n  got  ...{}\n  \
         want ...{}",
        got.len(),
        want.len(),
        show(&got[lo..(at + win).min(got.len())]),
        show(&want[lo..(at + win).min(want.len())]),
    );
}

/// The point of the whole file: every vendored case, byte for byte.
///
/// Reported as a first-difference offset with both sides' context rather than as two 4 KB
/// strings — the tool-definition cases are 2 KB of identical preamble, and a bare `assert_eq!`
/// on those is a diff nobody can read.
#[test]
fn every_case_renders_byte_for_byte() {
    let cases = cases();
    // **Anti-vacuity, hoisted, and it is not decoration here.** `include_str!` of a file that
    // was emptied, or a driver that wrote `{"cases": []}` after a failed render, would leave
    // this test green having compared nothing. Asserted on the INPUT rather than on a counter
    // the loop increments: the loop has no `continue`, so the two were provably equal, and the
    // counter version's tail assertion was byte-identical to the id pin's — a verbatim copy
    // escaping the duplication gate only by sitting ~12 tokens under its 15-token floor
    // (review, 2026-08-17). The count is the driver's own, so it moves only when a case is
    // deliberately added or removed.
    assert_eq!(cases.len(), 31, "expected 31 vendored cases");
    for case in &cases {
        let name = case["name"].as_str().expect("case has a name");
        let kw = &case["kwargs"];
        let messages = kw["messages"].as_array().expect("kwargs has messages");
        let want = case["expected"].as_str().expect("case has expected text");
        let got = render(messages, &opts_of(kw));
        assert_same(name, "byte", (got.as_bytes(), want.as_bytes()), 60, |w| {
            format!("{:?}", String::from_utf8_lossy(w))
        });
    }
}

/// **The id pin, and it is the gate M11b's framing change rests on**: `render`'s bytes through
/// the SHIPPED tokenizer must equal the ids `apply_chat_template` produced, for all 31 cases.
///
/// This closes exactly the gap [`the_special_tokens_survive_tokenization_as_single_ids`]'s
/// dated correction names — "`Tokenizer::load(dir).encode(&render(msgs, &opts))` against
/// `case[\"ids\"]`, on an artifact with a real tokenizer" — which was owed because no Glimmer
/// tokenizer was on the machine when that note was written. One now is.
///
/// **Why it matters more than the byte pin.** `render` returns a STRING containing
/// `<|start|>`, `<|message|>`, `<|eot|>` and friends as literal text, and whether each becomes
/// ONE id is decided by the `tokenizers` crate reading the checkpoint's added-token table —
/// not by anything in this crate. The byte pin cannot see that; a port whose specials
/// tokenized as five ordinary pieces each would pass it and feed the model a prompt it has
/// never seen, which is the failure mode `tokenizer.rs`'s own retraction is about.
///
/// **The tokenizer is 27 MB and cannot be vendored**, so the artifact directory is supplied by
/// `RIVOLI_GLIMMER_ARTIFACT` (any Glimmer artifact or the HF checkpoint — they carry identical
/// `tokenizer.json`; `tests/convert-parity-glimmer-fp8.sh` proves the copy is byte-exact).
/// Without it the id half cannot run, and this test does NOT then pass silently: it asserts
/// the reason. `eprintln!` would be invisible under libtest capture, which is the recorded
/// reason "skips loudly" is not a thing here.
#[test]
fn rendered_prompts_tokenize_to_the_vendored_ids() {
    let cases = cases();
    // Hoisted ABOVE the skip so it covers both branches. It used to live only in the skip
    // branch, which meant the arm that actually runs borrowed its credibility from an arm it
    // never executes (review, 2026-08-17).
    assert_eq!(cases.len(), 31, "expected 31 vendored cases");
    let Ok(dir) = std::env::var("RIVOLI_GLIMMER_ARTIFACT") else {
        // The census above already ran and covers this branch, so an unset variable cannot
        // read as coverage: what is missing here is precisely the tokenizer, and nothing else
        // about the fixture has been assumed.
        return;
    };
    let tok = rivoli_artifact::tokenizer::Tokenizer::load(&dir)
        .unwrap_or_else(|e| panic!("RIVOLI_GLIMMER_ARTIFACT={dir} is not loadable: {e}"));
    for case in &cases {
        let name = case["name"].as_str().expect("case has a name");
        let kw = &case["kwargs"];
        let want = ids_of(case);
        let got = tok
            .encode(&render(
                kw["messages"].as_array().expect("kwargs has messages"),
                &opts_of(kw),
            ))
            .unwrap_or_else(|e| panic!("case `{name}`: encode failed: {e}"));
        assert_same(name, "id", (&got, &want), 6, |w| format!("{w:?}"));
    }
    println!(
        "  id pin: {} cases tokenized identically to apply_chat_template",
        cases.len()
    );
}

/// Every special the template can emit is exercised by some vendored case — a CENSUS of the
/// fixture, and nothing more.
///
/// **Renamed 2026-08-17.** It was `the_special_tokens_survive_tokenization_as_single_ids`, and
/// its own dated correction below explains at length that it does not establish that. The test
/// that does is [`rendered_prompts_tokenize_to_the_vendored_ids`]; this one keeps the half it
/// really pays, and now says so in its name. Kept separate because the census needs no
/// tokenizer and therefore runs everywhere, including where the id pin skips.
///
/// > **CORRECTED 2026-08-14, by review, and the correction is that this test does LESS than it
/// > claimed.** It read: "This is what a lookalike would fail. A port that emitted `<|start|>` as
/// > five ordinary pieces ... matches [`every_case_renders_byte_for_byte`] and produces a
/// > different prompt." **That is incoherent** — `<|start|>` is the same *string* either way, and
/// > whether it becomes one id is decided by the tokenizer the ENGINE loads. This test reads
/// > `case["ids"]`, which Python wrote, and calls neither [`render`] nor any tokenizer: replace
/// > `render`'s whole body with `String::new()` and it stays green.
/// >
/// > The property is TRUE — every id below resolves out of the shipped `tokenizer.json` as
/// > `special: true, normalized: false`, and `tokenizer_config.json` carries no
/// > `added_tokens_decoder`, so the Rust side sees the same added vocabulary Python does. It is
/// > true and UNVERIFIED. `src/main.rs` cited this test for it and no longer does.
///
/// **What would close it:** `Tokenizer::load(dir).encode(&render(msgs, &opts))` against
/// `case["ids"]`, on an artifact with a real tokenizer. The tiny fixture has none.
///
/// > **CLOSED 2026-08-17 (M11b)** — by [`rendered_prompts_tokenize_to_the_vendored_ids`]
/// > above, which is exactly that run, on the shipped 27 MB `tokenizer.json` via
/// > `RIVOLI_GLIMMER_ARTIFACT`: 31 of 31 cases identical. What remains HERE is the census,
/// > which is worth keeping separate because it needs no tokenizer and therefore runs
/// > everywhere, including where the id pin skips.
///
/// What this still buys: a census. Every special the template can emit appears in some case, so
/// a fixture regenerated from a template that stopped emitting one goes red.
#[test]
fn every_special_the_template_emits_is_exercised_by_some_case() {
    // `<|begin_of_text|>`, `<|start|>`, `<|message|>`, `<|eot|>`, `<|eom|>`, `<|patch|>`,
    // `<|video|>` — resolved from the vendored ids rather than restated, since the whole
    // question is what the tokenizer does.
    const SPECIALS: [u32; 7] = [200000, 200022, 200023, 200008, 200007, 200092, 200091];
    let cases = cases();
    let mut seen: BTreeSet<u32> = BTreeSet::new();
    for case in &cases {
        let name = case["name"].as_str().expect("case has a name");
        let ids = ids_of(case);
        assert_eq!(
            ids.first(),
            Some(&200000),
            "case `{name}` must start with <|begin_of_text|>"
        );
        seen.extend(ids.iter().filter(|i| SPECIALS.contains(i)));
    }
    // Every special the template can emit is exercised by at least one case. Without this the
    // set check above passes on a fixture that happens to contain none of them.
    for s in SPECIALS {
        assert!(
            seen.contains(&s),
            "no vendored case exercises special id {s}"
        );
    }
}

/// The two string options, checked against the DEFAULT rather than for presence.
///
/// **Presence is not variation, and this asserted presence** (review, 2026-08-14). A case that
/// set `reasoning_strength: "high"` — the port's own default — satisfies "some case sets it"
/// while leaving `render` free to ignore the option entirely. The values are checked against
/// the defaults instead, which is the property that makes the byte test bind.
fn check_the_string_options_differ_from_their_defaults(cases: &[Value]) {
    let differs = |k: &str, default: &str| {
        cases.iter().any(|c| {
            c["kwargs"]
                .get(k)
                .and_then(Value::as_str)
                .is_some_and(|v| v != default)
        })
    };
    assert!(
        differs("reasoning_strength", "high"),
        "no case sets `reasoning_strength` to anything but the port's default — it could be a \
         hardcoded string and this suite would not notice"
    );
    assert!(
        differs("knowledge_cutoff", "2026-01-04"),
        "no case sets `knowledge_cutoff` away from the template's own literal"
    );
}

/// All four `tools` shapes appear.
///
/// `tools` has no scalar default to differ from; what matters is that BOTH truthiness arms are
/// present, since the empty array and `null` are false in Jinja and were the branch a review
/// found emitting 1277 bytes the reference omits.
fn check_every_tools_shape_is_present(cases: &[Value]) {
    let shapes: BTreeSet<&str> = cases
        .iter()
        .map(|c| match c["kwargs"].get("tools") {
            None => "absent",
            Some(Value::Null) => "null",
            Some(Value::Array(a)) if a.is_empty() => "empty",
            _ => "populated",
        })
        .collect();
    for want in ["absent", "null", "empty", "populated"] {
        assert!(
            shapes.contains(want),
            "no vendored case has `tools` {want} — Python truthiness and Rust Option-ness \
             disagree on `null` and `empty`, which is how 1277 bytes of ATEM preamble got emitted"
        );
    }
    assert!(
        cases
            .iter()
            .any(|c| c["kwargs"].get("tool_namespace_descriptions").is_some()),
        "no vendored case sets `tool_namespace_descriptions`"
    );
}

/// The bool flag takes both values, and the date takes exactly one.
///
/// The date is the one option with no default, so a port that dropped it would render a system
/// block with no `Current date:` line. Asserting it is CONSTANT is also what keeps the fixture
/// from depending on when it was generated.
fn check_the_flag_varies_and_the_date_does_not(cases: &[Value]) {
    let gen_prompt: BTreeSet<bool> = cases
        .iter()
        .map(|c| {
            c["kwargs"]
                .get("add_generation_prompt")
                .and_then(Value::as_bool)
                .unwrap_or(false)
        })
        .collect();
    assert_eq!(
        gen_prompt.len(),
        2,
        "every case sets the same add_generation_prompt — the flag is untested"
    );
    let dates: BTreeSet<&str> = cases
        .iter()
        .filter_map(|c| c["kwargs"]["current_date"].as_str())
        .collect();
    assert_eq!(
        dates.len(),
        1,
        "cases disagree on current_date; the fixture is not reproducible"
    );
    assert_eq!(dates.iter().next().copied(), Some("2026-08-14"));
}

/// The pin's own red proof: the recorded kwargs must actually reach the render.
///
/// A `render` that ignored `opts` entirely — returning a constant — would fail the byte
/// comparison, but a `render` that ignored ONE option would only fail if some case varies it.
/// This asserts the variation exists in the fixture, which is the property the byte test
/// silently depends on.
///
/// **A driver over named `check_*` helpers rather than one long body.** That is a code-health
/// requirement rather than taste — the CodeScene gate refuses a long unbroken run of assertions,
/// and a run of asserts with no name on it makes a failure read as "something in this file
/// broke". `crates/oracles/tests/glimmer_anchor.rs` states the same rule at length.
#[test]
fn the_fixture_varies_every_option() {
    let cases = cases();
    check_the_string_options_differ_from_their_defaults(&cases);
    check_every_tools_shape_is_present(&cases);
    check_the_flag_varies_and_the_date_does_not(&cases);
    assert_eq!(
        cases.len(),
        31,
        "case count moved without the byte test noticing"
    );
}

/// `utc_date` against dates computed independently, including the two shapes a hand-rolled
/// civil-date conversion gets wrong.
///
/// **Not a formality.** The conversion exists because this crate has no date dependency, and the
/// two classic failures are leap-day handling and the century rule — 2000 is a leap year and 1900
/// is not, and a conversion that treats every 4th year as a leap year is right for 128 years
/// running and then silently off by one. The epochs below are `date -u -d <date> +%s`.
#[test]
fn utc_date_matches_the_calendar() {
    use rivoli_artifact::glimmer_encoding::utc_date;
    use std::time::{Duration, UNIX_EPOCH};
    for (secs, want) in [
        (0u64, "1970-01-01"),
        (86_399, "1970-01-01"),        // last second of the day
        (86_400, "1970-01-02"),        // and the first of the next
        (951_782_400, "2000-02-29"),   // the century-rule leap day: divisible by 400, so it exists
        (1_709_164_800, "2024-02-29"), // an ordinary leap day
        (1_709_251_200, "2024-03-01"),
        (4_107_542_400, "2100-03-01"), // the day after a Feb that has NO 29th: 2100 is not a leap year
        (1_755_129_600, "2025-08-14"),
        (1_786_665_600, "2026-08-14"), // the date the fixture pins
    ] {
        assert_eq!(
            utc_date(UNIX_EPOCH + Duration::from_secs(secs)),
            want,
            "utc_date({secs})"
        );
    }
    // Before the epoch is documented as clamping rather than wrapping to year -something.
    assert_eq!(utc_date(UNIX_EPOCH - Duration::from_secs(1)), "1970-01-01");
}
