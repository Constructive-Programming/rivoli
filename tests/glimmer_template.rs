//! The chat-template byte pin: `artifact::glimmer_encoding::render` against Muse Glimmer's
//! own `chat_template.jinja`, rendered by the model's own tokenizer.
//!
//! **The expected side is not a reading of the template — it is the template's output.**
//! `tests/glimmer_template_driver.py` runs `AutoTokenizer.apply_chat_template` on
//! `meta-models/Muse-Glimmer-30B` over 31 cases and vendors `(kwargs, expected, ids)` into
//! `tests/glimmer-chat-cases.json`. That is what makes this a pin rather than a second
//! transcription: GLM's hand-port drifted to GLM-4's framing and survived months of review
//! because nothing ever compared it against the checkpoint's own file
//! (`artifact/tokenizer.rs`'s dated correction, and the `artifact-drops-the-chat-template`
//! note). `glimmer_reference.rs` makes the same argument for the decode loop.
//!
//! No GPU, no lock, no network, no Python — the fixture is bytes in the tree.
#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

use std::collections::BTreeSet;

use rivoli::artifact::glimmer_encoding::{GlimmerChatOpts, render};
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

/// The point of the whole file: every vendored case, byte for byte.
///
/// Reported as a first-difference offset with both sides' context rather than as two 4 KB
/// strings — the tool-definition cases are 2 KB of identical preamble, and a bare `assert_eq!`
/// on those is a diff nobody can read.
#[test]
fn every_case_renders_byte_for_byte() {
    let cases = cases();
    let mut checked = 0usize;
    for case in &cases {
        let name = case["name"].as_str().expect("case has a name");
        let kw = &case["kwargs"];
        let messages = kw["messages"].as_array().expect("kwargs has messages");
        let want = case["expected"].as_str().expect("case has expected text");
        let got = render(messages, &opts_of(kw));
        if got != want {
            let at = got
                .as_bytes()
                .iter()
                .zip(want.as_bytes())
                .position(|(a, b)| a != b)
                .unwrap_or(got.len().min(want.len()));
            let lo = at.saturating_sub(60);
            panic!(
                "case `{name}` diverges at byte {at} (got {} bytes, want {})\n  got  ...{:?}\n  want ...{:?}",
                got.len(),
                want.len(),
                &got[lo..(at + 60).min(got.len())],
                &want[lo..(at + 60).min(want.len())],
            );
        }
        checked += 1;
    }
    // **Anti-vacuity, and it is not decoration here.** `include_str!` of a file that was
    // emptied, or a driver that wrote `{"cases": []}` after a failed render, would leave this
    // test green having compared nothing. The count is the driver's own, so it moves only when
    // a case is deliberately added or removed.
    assert_eq!(
        checked, 31,
        "expected 31 vendored cases, compared {checked}"
    );
}

/// The vendored ids exist and every special the template can emit is exercised by some case.
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
/// What this still buys: a census. Every special the template can emit appears in some case, so
/// a fixture regenerated from a template that stopped emitting one goes red.
#[test]
fn the_special_tokens_survive_tokenization_as_single_ids() {
    // `<|begin_of_text|>`, `<|start|>`, `<|message|>`, `<|eot|>`, `<|eom|>`, `<|patch|>`,
    // `<|video|>` — resolved from the vendored ids rather than restated, since the whole
    // question is what the tokenizer does.
    const SPECIALS: [u32; 7] = [200000, 200022, 200023, 200008, 200007, 200092, 200091];
    let cases = cases();
    let mut seen: BTreeSet<u32> = BTreeSet::new();
    for case in &cases {
        let name = case["name"].as_str().expect("case has a name");
        let ids: Vec<u32> = case["ids"]
            .as_array()
            .expect("case has ids")
            .iter()
            .map(|v| v.as_u64().expect("id is a number") as u32)
            .collect();
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

/// The pin's own red proof: the recorded kwargs must actually reach the render.
///
/// A `render` that ignored `opts` entirely — returning a constant — would fail the byte
/// comparison, but a `render` that ignored ONE option would only fail if some case varies it.
/// This asserts the variation exists in the fixture, which is the property the byte test
/// silently depends on.
#[test]
fn the_fixture_varies_every_option() {
    let cases = cases();
    // **Presence is not variation, and this asserted presence** (review, 2026-08-14). A case
    // that set `reasoning_strength: "high"` — the port's own default — satisfies "some case sets
    // it" while leaving `render` free to ignore the option entirely. The values are checked
    // against the defaults instead, which is the property that makes the byte test bind.
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
    // `tools` has no scalar default to differ from; what matters is that BOTH truthiness arms are
    // present, since the empty array and `null` are false in Jinja and were the branch a review
    // found emitting 1277 bytes the reference omits.
    let tool_shapes: BTreeSet<&str> = cases
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
            tool_shapes.contains(want),
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
    // The date is the one option with no default, so a port that dropped it would render a
    // system block with no `Current date:` line. Cheap to assert that it is non-empty and
    // constant, which is also what keeps the fixture from depending on when it was generated.
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
    use rivoli::artifact::glimmer_encoding::utc_date;
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
