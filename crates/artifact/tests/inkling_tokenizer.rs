//! Inkling 2's tiktoken vocabulary, scored against FIRST-PARTY ids.
//!
//! **The expected side is `AutoTokenizer`'s own output, not a reading of the files.** The ids
//! below were measured with HuggingFace's tokenizer over the checkpoint at
//! `docs/measurement/inkling2-reference/` (the vendored `tiktoken.model` and
//! `mini/tokenizer_config.json` are compiled into this binary with `include_str!`, so this gate
//! asserts in CI and needs no artifact on disk).
//!
//! **What the cases are chosen to cover, since a tokenizer gate that only round-trips proves
//! nothing.** Four ordinary strings pin the BPE itself (`"Hi"`, a punctuation+space run,
//! four-space collapse — the `\s+(?!\S)` lookahead — and mixed Latin/CJK); one pins the
//! inline-specials shape (four control ids in a row); and one pins the chat framing by rendering
//! a real prompt through [`rivoli_artifact::inkling_encoding`] and comparing its ids. The
//! specials census walks `added_tokens_decoder` itself, so the count is derived from the file
//! rather than remembered.
//!
//! **The disagreement case is the red half of the cross-check.** `Vocab::load_parts` verifies
//! each config spelling against the model's bytes; a config whose spelling is perturbed must be
//! REFUSED, and the test mutates the vendored JSON to prove the refusal fires. Without it the
//! check is a claim that has only ever been green (P7).

// An integration test is a harness: a failed `expect` here IS the report.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use rivoli_artifact::inkling_encoding::{self, InklingChatOpts};
use rivoli_artifact::tiktoken::{SPECIAL_SLOTS, Vocab};
use serde_json::json;

/// The checkpoint's vocabulary — 200,058 lines, specials INCLUDED as base tokens.
const RANKS: &str = include_str!("../../../docs/measurement/inkling2-reference/tiktoken.model");
/// Mini's config, which names the 60 inline specials.
const CONFIG: &str =
    include_str!("../../../docs/measurement/inkling2-reference/mini/tokenizer_config.json");

fn vocab() -> Vocab {
    Vocab::load_parts(RANKS, CONFIG).expect("the vendored Inkling 2 vocabulary loads")
}

/// The inline shape: 200,058 ids total, no reserved slots above the last special.
#[test]
fn the_vocabulary_reports_exactly_the_measured_size() {
    let v = vocab();
    assert_eq!(v.n_vocab(), 200_058);
    assert_eq!(v.num_base(), 200_058);
    assert_eq!(v.special("<|message_user|>"), Some(200_000));
    assert_eq!(v.special("<|content_model_end_sampling|>"), Some(200_006));
    assert_eq!(v.special("<|end_message|>"), Some(200_010));
}

/// Every special the config names is a special here, and decodes back to its own spelling —
/// both ends read from `added_tokens_decoder`, so a dropped or shifted id cannot hide.
#[test]
fn every_added_token_is_a_special_and_decodes_to_itself() {
    let v = vocab();
    let doc: serde_json::Value = serde_json::from_str(CONFIG).expect("the config is JSON");
    let map = doc["added_tokens_decoder"]
        .as_object()
        .expect("the config names added tokens");
    assert_eq!(map.len(), 60, "the pinned config names 60 specials");
    for (key, entry) in map {
        let id: u32 = key.parse().expect("an added-token key is an id");
        let name = entry["content"]
            .as_str()
            .expect("an added token has a spelling");
        assert_eq!(v.special(name), Some(id), "special {name:?}");
        assert_eq!(v.decode(&[id]).unwrap(), name, "decode of {id}");
    }
}

/// The first-party ids, character for character.
#[test]
fn measured_strings_encode_to_their_first_party_ids() {
    let v = vocab();
    let cases: [(&str, &[u32]); 5] = [
        ("Hi", &[4896]),
        ("Hello, world!", &[32269, 11, 150770, 0]),
        ("a    b", &[64, 149862, 149963]),
        ("hello你好", &[32205, 11739, 23643]),
        (
            "<|message_user|><|content_text|>Hi<|end_message|>",
            &[200000, 200004, 4896, 200010],
        ),
    ];
    for (text, want) in cases {
        assert_eq!(v.encode(text).unwrap(), want, "encode({text:?})");
    }
}

/// The chat framing, end to end: render, tokenize, compare against `AutoTokenizer`'s ids for
/// the same conversation at `reasoning_effort = "none"`.
#[test]
fn the_rendered_prompt_encodes_to_the_measured_ids() {
    let v = vocab();
    let prompt = inkling_encoding::render(
        &[json!({"role": "user", "content": "Hi"})],
        &InklingChatOpts {
            reasoning_effort: Some("none"),
        },
    )
    .expect("a single user turn renders");
    assert_eq!(
        prompt,
        "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>\
         <|message_user|><|content_text|>Hi<|end_message|>"
    );
    assert_eq!(
        v.encode(&prompt).unwrap(),
        vec![
            200002, 200004, 34182, 279, 154554, 150628, 25, 220, 15, 200010, 200000, 200004, 4896,
            200010,
        ],
        "prompt was {prompt:?}"
    );
}

/// `decode` is the inverse on every case above, and refuses an id past the end.
#[test]
fn decode_round_trips_and_refuses_past_the_vocabulary() {
    let v = vocab();
    for text in [
        "Hi",
        "Hello, world!",
        "a    b",
        "hello你好",
        "<|message_user|><|content_text|>Hi<|end_message|>",
    ] {
        let ids = v.encode(text).unwrap();
        assert_eq!(v.decode(&ids).unwrap(), text, "round trip of {text:?}");
    }
    let err = v
        .decode(&[200_058])
        .expect_err("200058 is one past the end");
    assert!(
        err.to_string().contains("outside this vocabulary"),
        "unexpected refusal: {err}"
    );
}

/// **The cross-check's red proof.** A config that renames a special without the model agreeing
/// must be refused, not loaded with two answers for one id.
#[test]
fn a_config_that_disagrees_with_the_model_bytes_is_refused() {
    let mutated = CONFIG.replace("\"<|endoftext|>\"", "\"<|endoftext_x|>\"");
    assert_ne!(
        mutated, CONFIG,
        "the mutation must actually change the config"
    );
    let err = Vocab::load_parts(RANKS, &mutated)
        .err()
        .expect("the disagreement must be refused");
    let text = format!("{err:#}");
    assert!(
        text.contains("the two files disagree"),
        "unexpected refusal: {text}"
    );
}

/// **The detection rule, pinned.** A config naming nothing is the positional shape, so the same
/// model bytes would gain 256 reserved slots above them — the K3 reading, and the only one under
/// which unnamed slots stay spellable.
#[test]
fn an_empty_config_selects_the_positional_shape() {
    let v = Vocab::load_parts(RANKS, "{}").expect("an empty config still loads");
    assert_eq!(v.n_vocab(), 200_058 + SPECIAL_SLOTS);
    assert_eq!(v.num_base(), 200_058);
}
