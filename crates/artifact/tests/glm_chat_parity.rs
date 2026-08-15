//! Token-identity of the GLM chat surface against the pinned reference.
//!
//! **What this gate is for.** `Tokenizer::encode_chat_turns` is a hand-port of the
//! checkpoint's `chat_template.jinja` — there is no Jinja engine here — and the old tree
//! records that this exact port drifted onto GLM-4's `<|role|>\n` framing and stayed wrong
//! for MONTHS, invalidating every benchmark measured in that window. Nothing catches that
//! but ids compared against ids. A prose claim that the framing is right is not a gate.
//!
//! **Provenance of the expectations, both legs.** They are not round numbers and they were
//! not copied out of the new code.
//!
//! 1. *Recorded from the reference binary's own runs*, 2026-08-15, at the parity pin
//!    `wt/glimmer-s2` @ `6b7f496e` — the tree this module was ported from
//!    (`old:src/artifact/tokenizer.rs`), reading the same artifact directory this test
//!    reads.
//! 2. *Re-derived here from the artifact's own two files*, so leg 1 is decomposable rather
//!    than inherited. From `tokenizer.json`'s `added_tokens`: `[gMASK]` 154822, `<sop>`
//!    154824, `<|user|>` 154827, `<|assistant|>` 154828, `<think>` 154841, `</think>`
//!    154842, `<|endoftext|>` 154820, `<|observation|>` 154829. From its BPE vocab:
//!    `The` 785, `Ġsky` 12877, `Ġis` 374, `Ġblue` 6303, `Hi` 13041. From
//!    `generation_config.json`'s `eos_token_id`: `[154820, 154827, 154829]`.
//!
//! Every id below is therefore accounted for by name, and the *order* — prefix, turn
//! header, content, generation prompt — is the claim the ids are here to pin.
//!
//! **Skips when the artifact is absent**, matching `crates/engine/tests/
//! kernel_moe_artifact.rs`: the two files it needs are 19.3 MB of trained vocab plus a
//! 194-byte config, and the featureless CI job has neither. libtest captures the skip line,
//! so a skipped run looks green in the one-line summary — that is an honest limitation of
//! this pattern, not a claim that it "skips loudly"; the enforcement point is a run on this
//! box, whose evidence is `--nocapture`.

// tests: panic-on-failure is the idiom, and a broken gate should be loud.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use rivoli_artifact::tokenizer::{ChatOpts, Tokenizer};
use serde_json::{Value, json};

/// The shipped GLM-5.2 artifact. **The local path, not the `/swarm/storage/ai/rivoli`
/// NFS mirror** (byte-identical for these two files, checked 2026-08-16) — `tokenizer.json`
/// is 19.3 MB and this crate's tests are the featureless default `cargo test`, so pulling
/// it over the network on every run is a cost with nothing to buy.
const ART: &str = "/var/db/rivoli/glm52-vq3-full";

/// `(prompt, the ids the reference emits for it)`.
///
/// A table rather than two test functions because the claim is the same one twice, and the
/// second case exists to prove the first is not a coincidence of one tokenization: `Hi` is
/// a SINGLE content token, so the ten-id expectation minus its four content ids has to land
/// exactly on the seven-id one. Framing that is wrong by a token cannot satisfy both.
const CASES: &[(&str, &[u32])] = &[
    (
        "The sky is blue",
        &[
            154822, 154824, 154827, 785, 12877, 374, 6303, 154828, 154841, 154842,
        ],
    ),
    (
        "Hi",
        &[154822, 154824, 154827, 13041, 154828, 154841, 154842],
    ),
];

/// The artifact's stop tokens, as `generation_config.json` states them.
const EOS: &[u32] = &[154820, 154827, 154829];

/// Load the shipped tokenizer, or announce the skip and hand back `None`. One place, so
/// every test below has the same precondition and the same one-line skip.
fn load_or_skip(what: &str) -> Option<Tokenizer> {
    if std::fs::metadata(format!("{ART}/tokenizer.json")).is_err() {
        eprintln!("skip {what}: {ART}/tokenizer.json absent");
        return None;
    }
    Some(Tokenizer::load(ART).expect("load the shipped tokenizer"))
}

/// The framing, id for id. This is the gate the months-long `<|role|>\n` drift would have
/// failed on its first run.
#[test]
fn chat_framing_is_token_identical_to_the_reference() {
    let Some(tok) = load_or_skip("chat_framing") else {
        return;
    };
    // Anti-vacuity: a loop over an empty table passes forever, and the count is compared
    // against the table rather than against a literal, so adding a case cannot leave the
    // census behind.
    let mut checked = 0usize;
    for (text, want) in CASES {
        let got = tok.encode_chat(text).expect("encode_chat");
        assert_eq!(&got[..], *want, "encode_chat({text:?})");
        checked += 1;
        eprintln!("chat_framing OK {text:?} -> {got:?}");
    }
    assert_eq!(checked, CASES.len(), "compared {checked} of the cases");
    assert!(checked >= 2, "the case table lost its cross-check");
}

/// The stop tokens. Two of the three are TURN BOUNDARIES (`<|user|>`, `<|observation|>`),
/// which is why the framing above is load-bearing: outside a turn the model has no reason
/// to emit any of them and decode runs to its token limit.
#[test]
fn eos_ids_are_the_artifacts_own() {
    let Some(tok) = load_or_skip("eos_ids") else {
        return;
    };
    assert_eq!(tok.eos, EOS, "generation_config eos_token_id");
    eprintln!("eos_ids OK {:?}", tok.eos);
}

/// Sanity, not byte-parity: the ids the framing wraps still decode back to the prompt, so a
/// table that agreed on numbers while pointing at the wrong vocab entries could not pass.
#[test]
fn the_framed_ids_decode_back_to_the_prompt() {
    let Some(tok) = load_or_skip("decode_all") else {
        return;
    };
    let (text, ids) = CASES[0];
    let back = tok.decode_all(ids).expect("decode_all");
    assert!(
        back.contains(text),
        "decode_all({ids:?}) = {back:?}, which does not contain {text:?}"
    );
    eprintln!("decode_all OK {back:?}");
}

/// The framing the four assertions above do NOT reach — thinking, tool declarations,
/// assistant history and the continuation prefix — read back through `decode_all` as text.
///
/// **Why text and not ids here.** The four cases above pin ids because that is what parity
/// with the reference binary means. These pin the *template*, whose statement is in
/// `chat_template.jinja` and in the `encode_chat_turns` doc, and against which a rendered
/// string is the readable form; `decode_all` is itself pinned by the test above, and every
/// framing token in these strings appears in the ten-id expectation too. They exist because
/// the port introduced `Tokenizer::preamble` and `Tokenizer::turn_header` where the
/// reference had one straight-line function, and a refactor with no gate on the branches it
/// created is the thing this repo has already paid for once.
fn rendered(tok: &Tokenizer, turns: &[(&str, &str)], opts: &ChatOpts) -> String {
    let ids = tok
        .encode_chat_turns(turns, opts)
        .expect("encode_chat_turns");
    tok.decode_all(&ids).expect("decode_all")
}

/// [`rendered`] over the one-turn conversation the opts cases below all share, so each of
/// them states its `ChatOpts` and nothing else.
fn hi(tok: &Tokenizer, opts: &ChatOpts) -> String {
    rendered(tok, &[("user", "Hi")], opts)
}

/// Declared tools, thinking off — ONE spelling of this `ChatOpts` for both tool cases.
/// Written out twice they were a 47-token jscpd clone and `build.rs` refused them, which is
/// the right refusal: the cases differ only in the tools value, and a second spelling is a
/// place they could silently stop differing in just that one way. A `fn` rather than a
/// closure because a closure returning `ChatOpts<'_>` cannot tie the two lifetimes.
fn listed(tools: &Value) -> ChatOpts<'_> {
    ChatOpts {
        thinking: false,
        reasoning_effort: None,
        tools: Some(tools),
    }
}

/// Thinking leaves `<think>` OPEN and puts the effort turn before the conversation.
/// `capitalize` of the template's own default makes anything that is not "high" into "Max",
/// so `None` and `Some("medium")` must render identically — that is the template's
/// behaviour, not a shortcut, and it is the half a reimplementation gets wrong.
#[test]
fn thinking_opens_the_prefill_and_renders_the_effort_turn() {
    let Some(tok) = load_or_skip("thinking") else {
        return;
    };
    let think = |effort| ChatOpts {
        thinking: true,
        reasoning_effort: effort,
        tools: None,
    };
    let max = "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>Hi<|assistant|><think>";
    assert_eq!(hi(&tok, &think(None)), max);
    assert_eq!(
        hi(&tok, &think(Some("medium"))),
        max,
        "only high is not Max"
    );
    assert_eq!(
        hi(&tok, &think(Some("high"))),
        "[gMASK]<sop><|system|>Reasoning Effort: High<|user|>Hi<|assistant|><think>"
    );
    eprintln!("thinking OK {max}");
}

/// The tool declarations are their own `<|system|>` turn, AFTER any effort turn and BEFORE
/// the conversation — the order the template emits them in and therefore the order the model
/// saw. The turn's *contents* are gated deviceless in `tokenizer.rs`; what needs the real
/// vocab is that it lands in the right place and that an empty `tools` array emits nothing.
#[test]
fn the_tools_turn_sits_between_the_prefix_and_the_conversation() {
    let Some(tok) = load_or_skip("tools") else {
        return;
    };
    let decl = json!([{ "name": "read", "description": "read a file" }]);
    let with = hi(&tok, &listed(&decl));
    let tail = "<|user|>Hi<|assistant|><think></think>";
    assert!(
        with.starts_with("[gMASK]<sop><|system|>\n# Tools\n"),
        "{with}"
    );
    assert!(with.ends_with(tail), "{with}");
    assert!(
        with.contains("<tools>\n{\"name\": \"read\", \"description\": \"read a file\"}\n</tools>"),
        "{with}"
    );

    // An EMPTY array is not "tools" — the template's `{%- if tools -%}` is Python
    // truthiness, so it must render nothing at all rather than an empty `# Tools` turn.
    let none = json!([]);
    assert_eq!(hi(&tok, &listed(&none)), format!("[gMASK]<sop>{tail}"));
}

/// An assistant HISTORY turn carries a closed, empty `<think></think>` and its content
/// `.trim()`ed — the template's `content.strip()`. The trailing generation prompt is the
/// same shape, which is exactly why `encode_chat_continuation` can be derived from
/// `encode_chat` instead of re-emitting the framing.
#[test]
fn assistant_history_is_trimmed_and_carries_a_closed_think() {
    let Some(tok) = load_or_skip("history") else {
        return;
    };
    assert_eq!(
        rendered(
            &tok,
            &[("user", "Hi"), ("assistant", "  ok  "), ("user", "Hi")],
            &ChatOpts::default()
        ),
        "[gMASK]<sop><|user|>Hi<|assistant|><think></think>ok\
         <|user|>Hi<|assistant|><think></think>"
    );
}

/// The continuation is `encode_chat` minus the two-token conversation prefix, and nothing
/// else. Asserted against `encode_chat`'s own output rather than a second id list, because
/// a second list is the drift this function is derived to avoid.
#[test]
fn the_continuation_drops_exactly_the_conversation_prefix() {
    let Some(tok) = load_or_skip("continuation") else {
        return;
    };
    let (text, want) = CASES[1];
    let full = tok.encode_chat(text).expect("encode_chat");
    assert_eq!(&full[..2], &want[..2], "[gMASK] <sop>");
    assert_eq!(
        tok.encode_chat_continuation(text).expect("continuation"),
        full[2..],
        "the continuation is the same framing without the prefix"
    );
}
