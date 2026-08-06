//! `artifact::dsv4_encoding` against the four gold vectors the DeepSeek-V4-Flash checkpoint
//! ships in `encoding/tests/`, byte-for-byte.
//!
//! **This is the only test in the tree that can catch a wrongly-framed V4 prompt.** Nothing
//! downstream can: a prompt with the wrong turn markers still produces fluent text, and this
//! repo already lost months to exactly that on GLM's side (`encode_chat_turns` drifted onto
//! GLM-4's `<|role|>\n` framing and every benchmark before 2026-08-01 was measured off
//! template). The checkpoint's own `test_encoding_dsv4.py` is the executable specification;
//! this file is that specification, run against the Rust port.
//!
//! The gold is read from the checkpoint rather than copied into `tests/`, deliberately: a
//! vendored copy can be edited to make a failing port pass, and the whole value here is that
//! it cannot be. The unit tests inside `src/artifact/dsv4_encoding.rs` cover the paths these
//! four vectors miss and are hermetic, so a machine without the checkpoint is not left with
//! nothing.
#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

use rivoli::artifact::dsv4_encoding::{
    EncodeOpts, ParsedToolCall, ReasoningEffort, ThinkingMode, encode_messages,
    messages_from_openai, parse_message_from_completion_text,
};
use serde_json::Value;

/// Shared with the four V4 loading suites — including its rule that an explicitly-set env
/// var which does not resolve is a FAILURE and not a skip, because libtest hides stderr on
/// a passing test. `dead_code` because this is the first consumer that needs only two of
/// its three entry points.
#[path = "common/v4_artifact_dir.rs"]
#[allow(dead_code)]
mod v4_artifact_dir;

/// The checkpoint's `encoding/` folder, or `None` when this machine has no checkpoint.
///
/// ADV-11: `tests/v4_artifact.rs` reads the same `RIVOLI_V4_SRC` but SKIPS on a missing
/// weights index, where this FAILS on a missing `encoding/`. Not reconciled — that file is
/// not this change's to move — so the panic names which file wanted what.
fn encoding_dir() -> Option<String> {
    v4_artifact_dir::v4_artifact_at(
        "RIVOLI_V4_SRC",
        "/var/db/rivoli/deepseek-v4-flash-0731",
        "encoding/encoding_dsv4.py",
    )
    .map(|dir| format!("{dir}/encoding"))
}

/// An artifact whose `tokenizer.json` is DeepSeek-V4's. Any of them will do — the converter
/// copies it verbatim — so this takes the small three-layer fixture.
fn tokenizer() -> Option<rivoli::artifact::tokenizer::Tokenizer> {
    let dir = v4_artifact_dir::v4_artifact("tokenizer.json")?;
    Some(rivoli::artifact::tokenizer::Tokenizer::load(&dir).unwrap())
}

fn read(dir: &str, name: &str) -> String {
    std::fs::read_to_string(format!("{dir}/tests/{name}"))
        .unwrap_or_else(|e| panic!("{dir}/tests/{name}: {e}"))
}

fn gold(dir: &str, n: u8) -> String {
    read(dir, &format!("test_output_{n}.txt"))
}

fn input(dir: &str, n: u8) -> Value {
    serde_json::from_str(&read(dir, &format!("test_input_{n}.json"))).unwrap()
}

/// Encode the messages of `test_input_{n}.json` and diff against `test_output_{n}.txt`.
fn check(dir: &str, n: u8, messages: &[Value], thinking: ThinkingMode) -> String {
    let msgs = messages_from_openai(messages).unwrap();
    let got = encode_messages(msgs, &EncodeOpts::new(thinking)).unwrap();
    assert_eq!(got, gold(dir, n), "test_output_{n}.txt mismatch");
    got
}

/// Case 1 — thinking mode with tools: tool schemas in the system turn, a DSML tool call, a
/// tool result merged into a following user turn. `messages[0]["tools"] = tools` is the
/// reference test's own setup, not ours.
#[test]
fn case_1_thinking_with_tools() {
    let Some(dir) = encoding_dir() else { return };
    let td = input(&dir, 1);
    let mut messages = td["messages"].as_array().unwrap().clone();
    messages[0]["tools"] = td["tools"].clone();
    let prompt = check(&dir, 1, &messages, ThinkingMode::Thinking);

    // The reference's own round-trip: slice the two assistant turns back out of the prompt
    // it just built and parse them. This is what proves encode and parse are inverse — a
    // parser tested only on hand-written strings can agree with a wrong encoder.
    let turns = assistant_turns(&prompt);
    assert_eq!(turns.len(), 2, "{prompt:?}");
    let tc = parse_message_from_completion_text(turns[0], ThinkingMode::Thinking).unwrap();
    assert_eq!(
        tc.reasoning_content,
        "The user wants to know the weather in Beijing. I should use the get_weather tool."
    );
    assert_eq!(tc.content, "");
    assert_eq!(
        tc.tool_calls,
        vec![ParsedToolCall {
            name: "get_weather".into(),
            arguments: r#"{"location": "Beijing", "unit": "celsius"}"#.into(),
        }]
    );

    let fin = parse_message_from_completion_text(turns[1], ThinkingMode::Thinking).unwrap();
    assert_eq!(
        fin.reasoning_content,
        "Got the weather data. Let me format a nice response."
    );
    assert!(fin.content.contains("22°C"), "{:?}", fin.content);
    assert!(fin.tool_calls.is_empty());
}

/// Case 2 — thinking without tools: `drop_thinking` strips the earlier turn's reasoning.
#[test]
fn case_2_thinking_without_tools() {
    let Some(dir) = encoding_dir() else { return };
    let messages = input(&dir, 2);
    let prompt = check(
        &dir,
        2,
        messages.as_array().unwrap(),
        ThinkingMode::Thinking,
    );

    let turns = assistant_turns(&prompt);
    let parsed =
        parse_message_from_completion_text(turns[turns.len() - 1], ThinkingMode::Thinking).unwrap();
    assert_eq!(
        parsed.reasoning_content,
        "The user asks about the capital of France. It is Paris."
    );
    assert_eq!(parsed.content, "The capital of France is Paris.");
    assert!(parsed.tool_calls.is_empty());
    // The load-bearing half of `drop_thinking`: the FIRST turn's reasoning is gone.
    assert!(!prompt.contains("The user said hello"));
}

/// Case 3 — interleaved thinking + search: a `developer` turn carrying tools, a
/// `latest_reminder`, and CJK content that must survive `ensure_ascii=False` unescaped.
#[test]
fn case_3_developer_tools_and_reminder() {
    let Some(dir) = encoding_dir() else { return };
    let messages = input(&dir, 3);
    check(
        &dir,
        3,
        messages.as_array().unwrap(),
        ThinkingMode::Thinking,
    );
}

/// Case 4 — chat mode with a quick-instruction `action` task and a `latest_reminder`.
#[test]
fn case_4_quick_instruction_task() {
    let Some(dir) = encoding_dir() else { return };
    let messages = input(&dir, 4);
    check(&dir, 4, messages.as_array().unwrap(), ThinkingMode::Chat);
}

/// Every framing token must tokenize to ONE id.
///
/// `artifact::dsv4_encoding` produces a STRING, unlike GLM's `encode_chat_turns` which
/// assembles ids — so the whole port rests on the `tokenizers` crate splitting on
/// `added_tokens` inside ordinary text. If it did not, `<｜User｜>` would encode as a handful
/// of byte-BPE pieces the model has never seen in that position, the prompt would be off
/// template, and the output would still be fluent. That is precisely the silent failure this
/// module exists to prevent, so it gets its own gate rather than an assumption in a comment.
///
/// Host-only: `Tokenizer::load` reads two JSON files and touches no device.
#[test]
fn special_tokens_survive_the_tokenizer() {
    let Some(tok) = tokenizer() else { return };

    // Ids from the checkpoint's own `tokenizer.json` `added_tokens`, read 2026-08-05.
    for (text, id) in [
        ("<｜begin▁of▁sentence｜>", 0_u32),
        ("<｜end▁of▁sentence｜>", 1),
        ("<｜User｜>", 128803),
        ("<｜Assistant｜>", 128804),
        ("<think>", 128821),
        ("</think>", 128822),
        ("｜DSML｜", 128825),
        ("<｜latest_reminder｜>", 128828),
        ("<｜action｜>", 128829),
        ("<｜title｜>", 128836),
    ] {
        assert_eq!(tok.encode(text).unwrap(), vec![id], "{text} alone");
    }

    // …and in context, which is the case that actually matters: a token glued to prose on
    // both sides must still come out whole. Filtering rather than `contains` so ORDER and
    // multiplicity are pinned too — a tokenizer that emitted `<｜User｜>` twice, or after
    // `<｜Assistant｜>`, would satisfy a containment check and still be off template.
    let ids = tok
        .encode("<｜begin▁of▁sentence｜>S<｜User｜>hi<｜Assistant｜><think>")
        .unwrap();
    let framing = [0_u32, 128803, 128804, 128821];
    let got: Vec<u32> = ids
        .iter()
        .copied()
        .filter(|i| framing.contains(i))
        .collect();
    assert_eq!(got, framing, "{ids:?}");
    // `<tool_result>` is deliberately NOT an added token — the reference emits it from a
    // plain string template, so it must tokenize as several ordinary pieces.
    assert!(tok.encode("<tool_result>").unwrap().len() > 1);
}

/// Drive the reference and this port from the same corpus and diff the strings.
///
/// **A stronger gate than the four gold vectors, and it exists because the golds were not
/// enough.** Review on 2026-08-05 found a real divergence in
/// `sort_tool_results_by_call_order` — see the argument at that function — that all four
/// golds and sixteen hand-written unit tests passed straight over. Review then MEASURED how
/// thin that catch was, by monkey-patching the reference's own sort and re-encoding: of the
/// 600 RANDOM cases, **zero** discriminate either the index-base bug or the repeated-id bug
/// beside it. Random conversations do not reliably produce the shape.
///
/// So the corpus is SEEDED with hand-built adversarial conversations first, and the 600
/// random ones only explore around them — a guarantee where the shape is known to matter,
/// sampling for the combinations nobody thought of. Which seed pins what is recorded on each
/// one, so the next person to prune them can tell which are load-bearing. Both bugs are
/// re-confirmed to fail this test by mutation.
///
/// Host-only: one Python process, no device.
#[test]
fn differential_against_the_reference() {
    let Some(dir) = encoding_dir() else { return };
    let cases = generated_cases();

    let in_path = std::env::temp_dir().join(format!("rivoli-dsv4-{}.json", std::process::id()));
    std::fs::write(&in_path, serde_json::to_vec(&cases).unwrap()).unwrap();
    let driver = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/dsv4_reference_driver.py"
    );
    let run = std::process::Command::new("python3")
        .args([driver, &dir, in_path.to_str().unwrap()])
        .output();
    // ADV-8. No skip here, deliberately. `encoding_dir()` has already resolved, so the
    // checkpoint IS on this machine — and a green run that compared nothing is the vacuity
    // pattern this repo keeps re-learning (libtest hides stderr on a passing test, so the
    // `eprintln!` this used to do was invisible in an ordinary run). If the checkpoint is
    // here, the strongest gate in the tree runs or the suite goes red.
    let run = run.unwrap_or_else(|e| {
        panic!(
            "the checkpoint is present at {dir} but python3 will not run ({e}) — refusing to \
             pass by skipping the differential. Install python3 or move the checkpoint."
        )
    });
    // The input file is left behind ON FAILURE on purpose: the driver's docstring tells you
    // to re-run it by hand, and that needs the corpus that failed.
    assert!(
        run.status.success(),
        "reference driver failed: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    let want: Vec<Value> = serde_json::from_slice(&run.stdout).unwrap();
    std::fs::remove_file(&in_path).ok();

    assert_eq!(want.len(), cases.len());
    // A corpus the reference refuses would "agree" on every refusal and compare nothing.
    // It refuses NONE of these today, so this is exact rather than a slack bound: a case the
    // reference starts refusing is visible the day it happens instead of being absorbed.
    let encoded = want.iter().filter(|w| w["encoded"].is_string()).count();
    assert_eq!(
        encoded,
        cases.len(),
        "the reference refused {} cases — the corpus is garbage, not coverage",
        cases.len() - encoded
    );
    // BOUNDED: `assistant_turns` matches `<｜Assistant｜><think>`, which only the thinking
    // framing emits — so chat-mode cases and every `drop_thinking`-stripped turn contribute
    // nothing, and `parse_message_from_completion_text(_, Chat)` has no differential coverage
    // at all. The hermetic unit tests own that arm. Consistent on both sides, so a bounded
    // claim rather than a gap.
    //
    // Both directions are counted so neither can go quiet: the corpus currently encodes
    // 607/607 and yields 282 assistant turns, of which 75 PARSE on both sides and 207 are
    // refused by both (a trailing generation prompt and a `wo_eos` turn are unparseable by
    // design). A change that dropped the parsed count to zero would still "agree" everywhere.
    // Exact, and measured: the corpus is fixed-seed. 607 cases yield 282 assistant turns, of
    // which these parse on both sides and the rest are refused by both — a trailing generation
    // prompt and a `wo_eos` turn are unparseable by design, so agreeing on a refusal is the
    // common outcome and cannot be the only one.
    const PARSED_TURNS: usize = 75;
    let mut parsed_both = 0;
    for (case, want) in cases.iter().zip(&want) {
        let opts = EncodeOpts {
            thinking: match case["thinking"].as_str().unwrap() {
                "chat" => ThinkingMode::Chat,
                _ => ThinkingMode::Thinking,
            },
            drop_thinking: case["drop"].as_bool().unwrap(),
            add_bos: case["bos"].as_bool().unwrap(),
            reasoning_effort: match case["effort"].as_str().unwrap() {
                "high" => ReasoningEffort::High,
                "max" => ReasoningEffort::Max,
                _ => ReasoningEffort::Low,
            },
        };
        let got = messages_from_openai(case["messages"].as_array().unwrap())
            .and_then(|m| encode_messages(m, &opts));
        let prompt = match (want["encoded"].as_str(), got) {
            (Some(want), Ok(got)) => {
                assert_eq!(&got, want, "diverged on {case}");
                got
            }
            (None, Err(_)) => continue,
            (want, got) => panic!("reference {want:?} vs port {got:?} on {case}"),
        };

        // ADV-9: the same corpus, back through the PARSER. The reference sliced the assistant
        // turns out of the string it had just built and parsed each one; slice identically and
        // compare, so a parser that agrees with a WRONG encoder still fails.
        // The SPANS, not just their count. Both sides refuse 207 of 282 turns, and inside that
        // majority `null == Err` no matter what was sliced — so comparing counts alone let a
        // boundary divergence between the two `assistant_turns` hide completely. This turns
        // "kept in sync by comment" into a checked assertion.
        let turns = assistant_turns(&prompt);
        assert_eq!(
            serde_json::to_value(&turns).unwrap(),
            want["turns"],
            "the two assistant_turns disagree on {case}"
        );
        let want_parses = want["parses"].as_array().unwrap();
        for (turn, want) in turns.iter().zip(want_parses) {
            let got = parse_message_from_completion_text(turn, ThinkingMode::Thinking);
            match (want, got) {
                (Value::Null, Err(_)) => {}
                (want, Ok(got)) if !want.is_null() => {
                    // Compared as one value rather than field by field: a field added to
                    // `ParsedMessage` later is then diffed for free instead of silently
                    // skipped by three hand-written asserts.
                    let calls: Vec<Vec<String>> = got
                        .tool_calls
                        .iter()
                        .map(|t| vec![t.name.clone(), t.arguments.clone()])
                        .collect();
                    let got = serde_json::json!({
                        "content": got.content, "reasoning": got.reasoning_content,
                        "calls": calls,
                    });
                    assert_eq!(&got, want, "parse diverged on {turn:?}");
                    parsed_both += 1;
                }
                (want, got) => panic!("parse: reference {want:?} vs port {got:?} on {turn:?}"),
            }
        }
    }
    assert_eq!(
        parsed_both, PARSED_TURNS,
        "the parse half of this differential compared {parsed_both} turns, not {PARSED_TURNS} \
         — a fixed-seed corpus should not drift, and a slack bound here would hide a 50% \
         regression the way the encode gate above refuses to"
    );
}

/// Every completion-shaped span of `prompt`: after a thinking-mode assistant prefix, up to
/// the next user turn or the end.
///
/// One expression so that `assistant_turns` in `tests/dsv4_reference_driver.py` — which must
/// slice identically, or a mismatch is the slicing's fault and not the parser's — is
/// checkable against it at a glance rather than by reading two loops.
fn assistant_turns(prompt: &str) -> Vec<&str> {
    const MARKER: &str = "<｜Assistant｜><think>";
    prompt
        .split(MARKER)
        .skip(1)
        .map(|turn| turn.split("<｜User｜>").next().unwrap_or(turn))
        .collect()
}

/// A handful of hand-built adversarial conversations, then 600 from a fixed seed.
///
/// The seeds come first and are not optional: review measured that only 1 of the 600 random
/// cases discriminates the tool-result index base, so the shape this test was written for
/// hangs on one coin flip. These pin it regardless of the generator's mix, and — unlike the
/// unit tests that cover the same shapes — they are checked against the LIVE reference.
///
/// Deterministic so a failure is reproducible, and hand-rolled rather than drawn from a crate
/// because `rand` is not a dependency here.
fn generated_cases() -> Vec<Value> {
    // Numerical Recipes' LCG. Any full-period generator would do; this one is four lines.
    let mut state: u64 = 0x5f4d_cc3b_5aa7_65d6;
    let mut next = |n: usize| {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) as usize % n
    };

    let texts = [
        "hi",
        "",
        "多行\n文本",
        "a<b>c",
        "emoji 💡 ok",
        "tell me about 热海",
    ];
    let args = [
        r#"{"location": "Beijing"}"#,
        r#"{"q": "x", "n": 3}"#,
        r#"{"a": [1, 2], "o": {"k": "v"}, "b": true, "z": null}"#,
        r#"{"s": "line\nbreak", "u": "中文 <>&\""}"#,
        "not json at all",
        "{}",
    ];
    let tasks = ["action", "query", "authority", "domain", "read_url"];
    // NO NUMERIC LITERALS in this schema, and that is load-bearing rather than incidental:
    // a `minimum: 1e-5` here would turn the whole differential red for the known, accepted
    // float-formatting divergence in `dsv4_encoding`'s header, and the next person to widen
    // the generator would spend an afternoon on it.
    let tool = |name: &str| {
        serde_json::json!({"type": "function", "function": {"name": name, "description": "D",
            "parameters": {"type": "object", "properties": {"x": {"type": "string"}},
                           "required": ["x"]}}})
    };

    let call = |id: Option<&str>, name: &str| match id {
        Some(id) => serde_json::json!({"id": id, "type": "function",
            "function": {"name": name, "arguments": "{}"}}),
        None => serde_json::json!({"type": "function",
            "function": {"name": name, "arguments": "{}"}}),
    };
    let result = |id: &str| {
        serde_json::json!({"role": "tool", "tool_call_id": id,
                                               "content": format!("RES {id}")})
    };
    let seed_case = |msgs: Vec<Value>| {
        serde_json::json!({
            "messages": msgs, "thinking": "thinking", "effort": "low", "drop": true, "bos": true,
        })
    };
    let mut cases = vec![
        // PINS THE INDEX-BASE BUG (measured: this seed and the next are the only cases in
        // the corpus that do). An id-less call CONSUMES an index in the reference's
        // `enumerate`, so `c1` is key 1 and the unmatched result keeps the default 0.
        seed_case(vec![
            serde_json::json!({"role": "user", "content": "go"}),
            serde_json::json!({"role": "assistant", "content": "",
                "tool_calls": [call(None, "f"), call(Some("c1"), "g")]}),
            result("c1"),
            result("unmatched"),
        ]),
        // The index base again, further out. The result ORDER matters: with
        // `[c2, unmatched]` the correct sort (c2 -> key 2, unmatched -> default 0) swaps
        // them and the buggy one (c2 renumbered to 0) does not. Written the other way round
        // this seed was inert — measured in review, both algorithms produced the same bytes.
        seed_case(vec![
            serde_json::json!({"role": "user", "content": "go"}),
            serde_json::json!({"role": "assistant", "content": "",
                "tool_calls": [call(None, "f"), call(None, "g"), call(Some("c2"), "h")]}),
            result("c2"),
            result("unmatched"),
        ]),
        // PINS THE LAST-INDEX-WINS BUG, and is the only case in the corpus that does: the
        // reference assigns into a dict, so a repeated id keeps its LAST index. The random
        // corpus cannot produce this at all — generated ids are unique by construction.
        seed_case(vec![
            serde_json::json!({"role": "user", "content": "go"}),
            serde_json::json!({"role": "assistant", "content": "",
                "tool_calls": [call(Some("d"), "f"), call(Some("c"), "g"), call(Some("d"), "h")]}),
            result("d"),
            result("c"),
        ]),
        // `title` on an assistant turn: the only task the README puts on that role, and the
        // one that drives `prev_has_task` from the assistant side.
        seed_case(vec![
            serde_json::json!({"role": "user", "content": "P"}),
            serde_json::json!({"role": "assistant", "content": "A", "reasoning_content": "R",
                               "task": "title"}),
            serde_json::json!({"role": "assistant", "content": "T"}),
        ]),
        // A developer turn carrying tools (which turns `drop_thinking` off for the whole
        // conversation) and a `wo_eos` continuation. Both shapes were previously only
        // sampled by the random corpus; a seed pins them.
        seed_case(vec![
            serde_json::json!({"role": "system", "content": "S"}),
            serde_json::json!({"role": "developer", "content": "D", "tools": [
                {"type": "function", "function": {"name": "search", "description": "S",
                 "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}]}),
            serde_json::json!({"role": "assistant", "content": "A", "reasoning_content": "R"}),
            serde_json::json!({"role": "user", "content": "U2"}),
            serde_json::json!({"role": "assistant", "content": "partial",
                               "reasoning_content": "R2", "wo_eos": true}),
        ]),
        // The two shapes that render Python's `None` into the prompt — a null `tool` content
        // and a bare `latest_reminder`. Seeded rather than sampled because the random corpus
        // always supplies a content string, so it would never catch a regression on the one
        // byte change this port makes deliberately.
        seed_case(vec![
            serde_json::json!({"role": "latest_reminder"}),
            serde_json::json!({"role": "user", "content": "go"}),
            serde_json::json!({"role": "assistant", "content": "",
                               "tool_calls": [call(Some("c1"), "f")]}),
            serde_json::json!({"role": "tool", "tool_call_id": "c1", "content": null}),
        ]),
        // THE ACCEPT SIDE. Three shapes this port was, at some point in review, wrong to
        // REJECT — the direction a strict port fails in, and the one the over-refusal table
        // in `dsv4_encoding`'s header exists to bound. Seeding them means the differential
        // fails if any of the three narrowings is ever undone:
        //   * `tools` on a user turn — inert, because `merge_tool_messages` rebuilds the dict
        //   * `content: null` on an assistant turn — the ordinary OpenAI tool-call shape
        //   * junk `function.id` beside a good wrapper `id` — Python's `or` short-circuits
        seed_case(vec![
            serde_json::json!({"role": "user", "content": "U", "tools": [
                {"type": "function", "function": {"name": "f", "parameters": {}}}]}),
            serde_json::json!({"role": "assistant", "content": null, "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"id": 5, "name": "f", "arguments": "{}"}},
                {"id": "c2", "type": "function", "function": {"name": "g", "arguments": "{}"}}]}),
            result("c2"),
            result("c1"),
        ]),
    ];
    for c in 0..600 {
        let mut msgs: Vec<Value> = Vec::new();
        if next(10) < 9 {
            let mut sys =
                serde_json::json!({"role": "system", "content": texts[next(texts.len())]});
            match next(10) {
                0..=2 => sys["tools"] = serde_json::json!([tool("f"), tool("search")]),
                3 => sys["tools"] = serde_json::json!([]),
                4 => sys["response_format"] = serde_json::json!({"type": "json_object"}),
                _ => {}
            }
            msgs.push(sys);
        }
        if next(4) == 0 {
            msgs.push(serde_json::json!({"role": "latest_reminder", "content": "2026-02-21,广州"}));
        }
        // Track the ids the last assistant asked for, so tool results can be emitted out of
        // order against a real call list rather than against nothing.
        let mut pending: Vec<String> = Vec::new();
        for t in 0..(1 + next(5)) {
            // Weights out of 20: developer 2, tool results 3, orphan result 1, user 7,
            // assistant 7. The tool-result arm has a guard, so when nothing is pending its
            // three rolls fall through to the assistant arm — the mix shifts with history
            // rather than being fixed. That is exactly why the shapes that MATTER are seeded
            // above rather than sampled here.
            match next(20) {
                0..=1 => {
                    let mut d =
                        serde_json::json!({"role": "developer", "content": format!("D{t}")});
                    if next(2) == 0 {
                        d["tools"] = serde_json::json!([tool("search")]);
                    }
                    msgs.push(d);
                }
                2..=4 if !pending.is_empty() => {
                    // Results in reverse call order — the case the sort exists for.
                    for id in pending.drain(..).rev() {
                        msgs.push(serde_json::json!({"role": "tool", "tool_call_id": id,
                                                     "content": format!("RES {id}")}));
                    }
                }
                // Every arm of `tool_block_text` that the reference ACCEPTS, under the live
                // reference rather than only under the hermetic unit test: a plain text
                // block, a text block with no `text`, an unsupported named block, and the
                // two spellings (absent, null) of a `type` that renders the word `None`.
                5 => msgs.push(serde_json::json!({"role": "tool", "tool_call_id": "orphan",
                                                  "content": [{"type": "text", "text": "a"},
                                                              {"type": "text"},
                                                              {"type": "image"},
                                                              {},
                                                              {"type": null}]})),
                6..=12 => {
                    let mut u =
                        serde_json::json!({"role": "user", "content": texts[next(texts.len())]});
                    if next(5) == 0 {
                        u["task"] = Value::String(tasks[next(tasks.len())].into());
                    }
                    msgs.push(u);
                }
                _ => {
                    let mut a = serde_json::json!({"role": "assistant",
                        "content": texts[next(texts.len())], "reasoning_content": format!("R{t}")});
                    if next(3) == 0 {
                        let calls: Vec<Value> = (0..1 + next(2))
                            .map(|k| {
                                let f = serde_json::json!({"name": "f",
                                    "arguments": args[next(args.len())]});
                                // Every third call carries NO id — the shape that exposed the
                                // index-base bug, and the one the gold vectors never produce.
                                if next(3) == 0 {
                                    serde_json::json!({"type": "function", "function": f})
                                } else {
                                    let id = format!("c{c}_{t}_{k}");
                                    pending.push(id.clone());
                                    serde_json::json!({"id": id, "type": "function", "function": f})
                                }
                            })
                            .collect();
                        a["tool_calls"] = Value::Array(calls);
                    }
                    if next(12) == 0 {
                        a["wo_eos"] = Value::Bool(true);
                    }
                    // `title` is the one task the README puts on an assistant turn.
                    if next(15) == 0 {
                        a["task"] = Value::String("title".into());
                    }
                    msgs.push(a);
                }
            }
        }
        let thinking = ["chat", "thinking"][next(2)];
        let effort = ["low", "high", "max"][next(3)];
        let (drop, bos) = (next(2) == 0, next(4) != 0);
        cases.push(serde_json::json!({
            "messages": msgs, "thinking": thinking, "effort": effort, "drop": drop, "bos": bos,
        }));
    }
    cases
}
