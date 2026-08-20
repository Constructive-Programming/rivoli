//! Muse Glimmer's half of the OpenAI door: the request's prompt string and the reply's
//! channels.
//!
//! **Split out of [`super::oai`] at M11b**, which the two halves pushed past this repo's
//! 800-line soft cap. The cut is by MODEL rather than by direction on purpose: `oai.rs` is
//! GLM-shaped end to end — `messages_to_turns` flattens into GLM's turns, `parse_tool_calls`
//! reads GLM's `<tool_call>` markup, `split_think` reads GLM's `</think>` — and mixing a
//! second checkpoint's framing into it is how the two come to be read as one. The dispatch
//! that pairs them lives in `super::frame_prompt` and `super::split_channels`, and is gated
//! there.
//!
//! Pure functions of a request body and a generated string: no tokenizer, no engine, no
//! device. That is what lets the two decisions rivoli INVENTS here — the `developer` rewrite
//! and the reasoning-strength mapping — be gated at all. `super::frame_prompt` cannot be: it
//! needs a real 27 MB `Tokenizer`, which is why the id pin that covers THAT step lives in
//! `crates/artifact/tests/glimmer_template.rs` behind `RIVOLI_GLIMMER_ARTIFACT`.
//!
//! > **The reason stated here until review 2026-08-17 was FALSE** — "`frame_prompt` takes a
//! > `&Ctx`, which holds `&mut Engine`, so nothing under it is reachable without a GPU". It
//! > takes no `Ctx`, `super::split_channels` takes none either, and `serve/mod.rs` already
//! > hosts a deviceless `mod tests`. The consequence of believing it was real: the arch
//! > DISPATCH — the exact pairing whose mismatch caused M11b's revert — went ungated while
//! > both of its leaves were covered. `the_reply_is_read_back_with_the_same_template_that_framed_it`
//! > closes that, and it is in `serve/mod.rs` precisely because nothing stopped it being there.

use anyhow::{Context, Result};
use serde_json::Value;

/// Muse Glimmer's prompt STRING for one request — the two decisions serve makes that
/// `glimmer_encoding::render` cannot make for itself.
///
/// **Here rather than in `mod.rs` because a claim that cannot go red is not a claim (P7).**
/// `serve::frame_prompt` takes a `&Ctx`, which holds `&mut Engine`, so nothing under it is
/// reachable without a GPU. This module's header says it is where the pure functions and the
/// tests live; these two decisions belong to it, and they are tested below.
///
/// **1. `developer` becomes `system`.** Glimmer's template has no `developer` arm and its
/// `if/elif` chain has no `else`, so an unmapped one renders as NOTHING — the request's
/// instructions dropped, with a 200 on top. The array is cloned because `render` takes
/// `&[Value]`, and that signature is byte-pinned across 31 vendored cases.
///
/// **2. The reasoning strength, which is rivoli's mapping and not the template's.** The Jinja
/// has no thinking BOOLEAN: it always renders `Reasoning strength: <s>.`, defaulting to
/// `"high"`, with no arm that omits it. An explicit `reasoning_effort` passes through verbatim
/// — it is already a strength word, and `"none"` is what OpenAI clients send to mean "don't" —
/// and when the request names none, the server's `--think` picks between the template's
/// default and `"none"`. **That second half is invented here**; the model may still reason, and
/// no rendered string can prevent it. Said plainly because a reader comparing this against
/// `chat_template.jinja` will not find it there.
///
/// **`think == false` OUTRANKS an explicit effort, decided rather than fallen into** (review,
/// 2026-08-17). `thinking_mode` lets `enable_thinking: false` win over `reasoning_effort`, so
/// `{"enable_thinking": false, "reasoning_effort": "high"}` arrives as `(false, Some("high"))`.
/// The first draft read `effort.or(…)`, which rendered `high` — the opposite of what GLM's arm
/// does with the same body, where `thinking: false` closes `<think>` and the effort is
/// decoration. The specific instruction is "do not think"; a strength word is what to do IF
/// thinking. The test pins all five cells so this stays a decision.
///
/// **What the `developer` rewrite COSTS, stated because it is invisible otherwise.** `render`
/// emits its synthesised system block — the persona line, `Knowledge cutoff:` and
/// `Current date:` — only when NO message has `role == "system"`. Rewriting `developer` to
/// `system` therefore suppresses it, so a client using OpenAI's newer role name gets a prompt
/// with no date, even though `serve::frame_prompt` computes one for every request. That is the
/// same thing that happens when a client sends a real `system` turn, which is the template's
/// own designed behaviour — so the rewrite makes `developer` behave exactly like `system`,
/// which is what it is. Pinned below rather than left as prose.
///
/// **`tools` is deliberately NOT forwarded, and that is a scope line, not an oversight.**
/// `GlimmerChatOpts::tools` renders the ATEM preamble — `<atem:function_calls>` blocks — while
/// [`parse_tool_calls`] below reads GLM's `<tool_call>` markup and nothing else. Advertising a
/// call syntax this server cannot read back would turn every tool use into prose with
/// `finish_reason: "stop"`: a confidently wrong answer, which is worse than a model that does
/// not call tools. Closing it means porting an ATEM parser and the `to=<ns>.<fn>` recipient
/// convention beside it; until then a Glimmer tool request gets an honest untooled reply.
pub fn glimmer_prompt(
    body: &Value,
    (think, effort): (bool, Option<&str>),
    date: &str,
) -> Result<String> {
    use rivoli_artifact::glimmer_encoding::{GlimmerChatOpts, render};
    let mut messages = body
        .get("messages")
        .and_then(Value::as_array)
        .context("`messages` must be an array")?
        .clone();
    for m in &mut messages {
        if m.get("role").and_then(Value::as_str) == Some("developer") {
            m["role"] = Value::String("system".into());
        }
    }
    Ok(render(
        &messages,
        &GlimmerChatOpts {
            reasoning_strength: if think { effort } else { Some("none") },
            ..GlimmerChatOpts::new(date)
        },
    ))
}

/// Drop a trailing fragment that could still grow into one of Glimmer's turn markers.
///
/// **The streaming twin of [`delta`]'s U+FFFD rule, and found by the same kind of gate.** A
/// decoded PREFIX can end mid-marker — `"The sky is blue.<"` is three tokens away from
/// `"The sky is blue.<|eot|>"` — and emitting that `<` puts a character in the stream that the
/// finished text does not contain, which wedges `delta`'s `strip_prefix` for the rest of the
/// request exactly as a leaked header would. Holding it back costs one token of latency and
/// the next decode supersedes it.
///
/// Every marker starts `<|`, so the last `'<'` is the only place a fragment can begin.
fn trim_partial_marker(s: &str) -> &str {
    const MARKERS: [&str; 4] = ["<|message|>", "<|eot|>", "<|eom|>", "<|start|>"];
    match s.rfind('<') {
        Some(i) if MARKERS.iter().any(|m| m.starts_with(&s[i..])) => &s[..i],
        _ => s,
    }
}

/// Split a Muse Glimmer generation into `(reasoning, content)`.
///
/// **The channel is IN THE TEXT here, which is why this takes no `thinking` flag and
/// [`split_think`] does.** GLM's prompt ends at an open `<think>` and the mode decides how to
/// read the reply; Glimmer's prompt ends at `<|start|>assistant` and the MODEL names each
/// turn's recipient — `to=self` is the reasoning channel, anything else is the answer. A flag
/// here would be a second authority on a fact the bytes already carry.
///
/// Turn shape: `[<|start|>assistant] to=<recipient><|message|><body><|eot|>|<|eom|>`. The first
/// turn's `<|start|>assistant` is in the PROMPT, not the generation, which is why splitting on
/// it and treating every piece the same works.
///
/// **`complete` says whether `full` is the WHOLE generation, and it is load-bearing.** With no
/// `<|message|>` anywhere, a complete generation comes back whole as content — the
/// unstructured case, where dropping it would turn a short reply into an empty one, and an
/// empty `content` reads as a working server returning nothing. A PREFIX with no `<|message|>`
/// yet is the opposite situation and gets `("", "")`.
///
/// > **P0 caught by review 2026-08-17, before it shipped.** The flag did not exist and the
/// > fallback fired on both. `render` ends the prompt at `<|start|>assistant`, so the first
/// > tokens a model emits are the turn header ` to=user` — several tokens BEFORE `<|message|>`.
/// > `stream_decode` re-splits the prefix every token and sends `delta(sent, content)`, so
/// > those steps streamed the raw header to the client; then `<|message|>` arrived, `content`
/// > collapsed to the message body, and `strip_prefix(" to=user")` failed on it and on every
/// > token after — `sent_c` frozen forever. **Every streamed Glimmer reply would have been the
/// > literal text ` to=user` and nothing else, with `finish_reason: "stop"`.** The
/// > non-streaming path was correct throughout, and §6c's owed device recipe is a
/// > non-streaming `curl`, so neither would have caught it. A whole-text fallback is a
/// > property of a finished generation; asking a prefix to satisfy it was the error.
///
/// A reply truncated mid-reasoning has a `to=self` turn and no answer turn, so it reports all
/// reasoning and empty content — deliberately, on [`split_think`]'s own argument: presenting a
/// half-finished train of thought as the answer is worse than saying there is no answer yet.
pub fn split_glimmer(full: &str, complete: bool) -> (String, String) {
    let (mut reasoning, mut content) = (String::new(), String::new());
    let mut saw_message = false;
    for turn in full.split("<|start|>assistant") {
        let Some((header, body)) = turn.split_once("<|message|>") else {
            continue;
        };
        saw_message = true;
        // EARLIEST terminator wins, not `<|eot|>` preferentially. Both close a turn and only
        // one should appear in a well-formed one, but "try eot, else eom" quietly keeps an
        // `<|eom|>` and everything after it when a generation emits both — and a reply that
        // renders `<|eom|>` to the user is the leak this whole function exists to stop.
        let end = ["<|eot|>", "<|eom|>"]
            .iter()
            .filter_map(|m| body.find(m))
            .min()
            .unwrap_or(body.len());
        // EXACT, not `contains`. The renderer emits `<|start|>assistant to=self<|message|>`
        // for the reasoning channel, so the header is precisely `to=self` once trimmed; a
        // substring test would also claim a recipient like `to=selfcheck.run`, quietly filing
        // a tool call as reasoning and dropping it from `content`.
        let into = if header.trim() == "to=self" {
            &mut reasoning
        } else {
            &mut content
        };
        // On a PREFIX the tail may be a marker still arriving; see `trim_partial_marker`.
        let body = &body[..end];
        into.push_str(if complete {
            body
        } else {
            trim_partial_marker(body)
        });
    }
    if saw_message || !complete {
        (reasoning, content)
    } else {
        (String::new(), full.to_string())
    }
}

// **Named imports and an INNER allow, where every sibling test module opens with
// `#[allow(...)] mod tests { use super::*; use …;`.** That four-line preamble is one token run
// to jscpd, which normalizes identifiers — it reported this file against `http.rs` at 29
// tokens the moment the module was created (2026-08-17). An import list is the one duplication
// Rust gives you no way to factor, so the fix is to have a different one rather than an
// exemption claiming the copy is the point.
#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)] // a harness that cannot parse its own fixture should panic

    use serde_json::json;

    use super::{glimmer_prompt, split_glimmer};

    /// The `developer` role reaches the model. Glimmer's template drops an unrecognised role
    /// SILENTLY (no `else` in its chain), so without the rewrite this content vanishes behind
    /// a 200 — the one failure a server must never have.
    #[test]
    fn a_developer_turn_is_framed_as_a_system_turn_and_not_dropped() {
        let body = json!({"messages": [
            {"role": "developer", "content": "SENTINEL-DEV"},
            {"role": "user", "content": "hi"}
        ]});
        let got = glimmer_prompt(&body, (true, None), "2026-01-02").unwrap();
        assert!(
            got.contains("<|start|>system<|message|>SENTINEL-DEV"),
            "the developer turn did not reach the prompt as a system turn:\n{got}"
        );
        // **And what that COSTS, pinned so it cannot change silently.** Becoming a `system`
        // turn suppresses `render`'s synthesised block, so the date this server computes for
        // every request is not in the prompt. Identical to sending a real `system` turn — the
        // template's own behaviour — but it is the reason `Current date` can go missing.
        assert!(
            !got.contains("Current date:"),
            "the synthesised system block reappeared; if that is now intended, this \
             assertion is the thing to change:\n{got}"
        );
        let plain = glimmer_prompt(
            &json!({"messages": [{"role": "user", "content": "hi"}]}),
            (true, None),
            "2026-01-02",
        )
        .unwrap();
        assert!(
            plain.contains("Current date: 2026-01-02."),
            "without a system-shaped turn the date must be there, or this test proves nothing"
        );
    }

    /// The reasoning-strength mapping, all four cells. Two of them are rivoli's own invention
    /// on a template with no thinking boolean, which is exactly why they are pinned here and
    /// not left to a GPU round-trip that only asserts a reply terminates.
    #[test]
    fn the_reasoning_strength_maps_the_request_over_the_servers_default() {
        let body = json!({"messages": [{"role": "user", "content": "hi"}]});
        let of = |t, e| glimmer_prompt(&body, (t, e), "2026-01-02").unwrap();
        // An explicit effort renders verbatim WHEN thinking is on.
        assert!(of(true, Some("high")).contains("Reasoning strength: high."));
        assert!(of(true, Some("none")).contains("Reasoning strength: none."));
        // ...and `enable_thinking: false` outranks it, matching what GLM's arm does with the
        // same body. This cell is the one a `effort.or(...)` spelling gets backwards.
        assert!(of(false, Some("high")).contains("Reasoning strength: none."));
        // With none named, `--think` decides: on leaves the template's own default...
        assert!(of(true, None).contains("Reasoning strength: high."));
        // ...off asks for "none", the half no reference renderer would produce.
        assert!(of(false, None).contains("Reasoning strength: none."));
    }

    /// No `messages` array is a 400, not an empty conversation. `render` would happily emit a
    /// system block and a bare generation prompt, and the model would answer nobody.
    #[test]
    fn a_body_without_messages_is_refused() {
        assert!(glimmer_prompt(&json!({}), (true, None), "2026-01-02").is_err());
        assert!(glimmer_prompt(&json!({"messages": 7}), (true, None), "2026-01-02").is_err());
    }

    /// **The streaming property, and the gate that would have caught M11b's P0.**
    ///
    /// `stream_decode` re-splits the whole decoded prefix on every token and sends only what
    /// `delta` says is new, which is `strip_prefix` — so each channel must grow MONOTONICALLY
    /// as the generation arrives. If a prefix ever reports content the finished generation
    /// does not start with, that channel is wedged for the rest of the request: `delta`
    /// returns `None` from then on and the client never receives another byte of it.
    ///
    /// Simulated over every byte prefix rather than asserted on a handful of snapshots,
    /// because the failure lived in the first three tokens — before `<|message|>` arrives —
    /// which is exactly the region a hand-written case list does not think to cover.
    #[test]
    fn every_prefix_of_a_generation_grows_both_channels_monotonically() {
        for g in [
            " to=user<|message|>The sky is blue.<|eot|>",
            " to=self<|message|>scattering<|eom|><|start|>assistant to=user             <|message|>The sky is blue.<|eot|>",
        ] {
            let (want_r, want_c) = split_glimmer(g, true);
            // Byte prefixes, not char — `delta` works on bytes and the fixture is ASCII.
            for i in 0..=g.len() {
                let (r, c) = split_glimmer(&g[..i], false);
                assert!(
                    want_r.starts_with(&r),
                    "prefix {i} reported reasoning {r:?}, which the finished {want_r:?} does                      not start with — the stream is wedged from here on"
                );
                assert!(
                    want_c.starts_with(&c),
                    "prefix {i} reported content {c:?}, which the finished {want_c:?} does                      not start with — the stream is wedged from here on"
                );
            }
        }
    }

    /// The reply channels, read off the recipient the MODEL names rather than off a flag.
    #[test]
    fn split_glimmer_reads_the_recipient_and_never_swallows_a_reply() {
        // Answer only — the prompt supplied the leading `<|start|>assistant`.
        assert_eq!(
            split_glimmer(" to=user<|message|>The sky is blue.<|eot|>", true),
            (String::new(), "The sky is blue.".into())
        );
        // Reasoning turn, then the answer turn.
        assert_eq!(
            split_glimmer(
                " to=self<|message|>scattering<|eom|><|start|>assistant to=user\
                 <|message|>The sky is blue.<|eot|>",
                true
            ),
            ("scattering".into(), "The sky is blue.".into())
        );
        // Truncated mid-reasoning: all reasoning, no answer — `split_think`'s own rule.
        assert_eq!(
            split_glimmer(" to=self<|message|>scatter", true),
            ("scatter".into(), String::new())
        );
        // **The one that matters most.** No markers at all must not yield an empty reply:
        // an empty `content` reads as a working server returning nothing.
        assert_eq!(
            split_glimmer("bare text", true),
            (String::new(), "bare text".into())
        );
        assert_eq!(split_glimmer("", true), (String::new(), String::new()));
        // Truncated INSIDE the header — the non-streaming twin of the P0. A finished
        // generation this shape is genuinely unstructured, so the fragment is surfaced rather
        // than swallowed; a PREFIX of the same bytes is not finished and reports nothing.
        assert_eq!(
            split_glimmer(" to=us", true),
            (String::new(), " to=us".into())
        );
        assert_eq!(
            split_glimmer(" to=us", false),
            (String::new(), String::new())
        );
        // A recipient that merely STARTS with `self` is a tool call, not the reasoning
        // channel: it must land in content, where a caller can see it, not vanish into
        // `reasoning_content`.
        assert_eq!(
            split_glimmer(" to=selfcheck.run<|message|>CALL<|eot|>", true),
            (String::new(), "CALL".into())
        );
        // Both terminators present: the EARLIEST closes the turn, so nothing after it — and
        // no marker itself — reaches the user.
        assert_eq!(
            split_glimmer(" to=user<|message|>A<|eom|>B<|eot|>C", true),
            (String::new(), "A".into())
        );
    }
}
