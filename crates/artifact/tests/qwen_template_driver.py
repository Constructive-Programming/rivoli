#!/usr/bin/env python3
"""Vendor Qwen3.8-Flash-Next's OWN chat template output, so the Rust hand-port is pinned
against the real renderer rather than against a second transcription.

`docs/investigations/qwen-flash-next-port.md`'s exit-gate table asks track D for "id-pinned
template cases scored against the reference tokenizer's own output (gate G5)". This is the
*pin*: it renders the checkpoint's `chat_template.jinja` through
`AutoTokenizer.apply_chat_template` -- the same Jinja sandbox, the same `trim_blocks`, the same
`tojson` -- and writes `(name, kwargs, expected, ids)` quadruples to `qwen-chat-cases.json`
beside itself. `qwen_template.rs`, also beside it, reads them with no network, no GPU and no
Python.

**Why not pin against a literal written by hand.** GLM's template drifted for months to GLM-4's
`<|role|>\\n` framing (memory: `artifact-drops-the-chat-template`), and the reason it went
unnoticed is that the only thing it was ever checked against was a second reading by the same
author. A pin whose expected side comes from the model's own file cannot share a misreading
with the port.

**The negative cases are half the value.** This template calls `raise_exception` in eight
places and three of them are reachable from an ordinary OpenAI client (a `developer` role, a
conversation with no real user turn, a system turn that is not first). Those cases record the
exception MESSAGE, and the Rust side asserts its own refusal carries the same text -- so the
port is pinned in the reject direction too, which is the direction a strict port is most likely
to get wrong (`v4_encoding`'s `tools_on_a_role_that_cannot_render_them` is the recorded scar).

**Two shapes cannot be pinned here and are recorded rather than claimed.** `messages=[]` and a
MAPPING `tools` are both refused by transformers itself before the template is entered
(`ValueError: Cannot apply chat template to an empty conversation.` and `ValueError: Tools
should either be a JSON schema, ...`), so `apply_chat_template` has no opinion to record about
what the template would have done. The port reproduces the template's own text for the first
and the template's own else-branch for the second; neither is scored by this fixture.

Every case is a pure function of its inputs: unlike Glimmer's template this one calls no
`strftime_now`, so nothing here goes red at midnight and no date needs pinning.

Usage (no GPU, no lock, no wrapper script -- one command, seconds):

    HF_HUB_OFFLINE=1 QWEN_CKPT=/var/cache/rivoli/scratch/qwen-template/ckpt \\
    /var/cache/rivoli/scratch/qwen-template/venv/bin/python \\
        crates/artifact/tests/qwen_template_driver.py

The checkpoint directory needs only `tokenizer.json`, `tokenizer_config.json` and
`chat_template.jinja` -- 12.84 MB of a 172.76 GiB repo, so regenerating this does not need the
weights. Fetch them from `resolve/<REVISION>/<file>`, never from `/main/`. Measured versions
are recorded in the output's `provenance` block and asserted by the Rust side.
"""

import json
import os
import sys

# ---------------------------------------------------------------------------------------
# Provenance. Recorded INTO the fixture so the Rust side can gate on it: a pin checked only
# against a frozen copy of itself is decoration, so `qwen_template.rs` recomputes both hashes
# from `docs/measurement/qwen-reference/chat_template.jinja` and compares them to these.
# ---------------------------------------------------------------------------------------
REPO = "Qwen/Qwen3.8-Flash-Next-FP8"
REVISION = "236dfdf285828023ca3bcd3f37366c58a3469b13"
# `chat_template.jinja`, 8952 B. The SAME 8952 bytes are `tokenizer_config.json`'s
# `chat_template` key -- EXACTLY byte-identical, not identical-modulo-a-trailing-newline: the
# .jinja file does not end with one. Both homes were compared byte-for-byte 2026-08-31; the
# plan of record's census item 2 says "identical modulo the file's trailing newline", and the
# measured answer is that there is no modulo.
TEMPLATE_SHA256 = "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"
TEMPLATE_FNV1A64 = "8b6b0871c5db260b"
TEMPLATE_BYTES = 8952
TOKENIZER_SHA256 = "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"
TOKENIZER_CONFIG_SHA256 = "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27"

U = {"role": "user", "content": "Hi"}

# A real OpenAI tool schema. Rendered through `tojson` as-is -- this template does NOT unwrap
# the `{"type","function":{...}}` envelope the way GLM's does, so the envelope is part of the
# pinned bytes and a port that unwrapped it would be wrong by two keys per tool.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            # `<` `>` `&` `'` decide whether `tojson` is Jinja's HTML-safe one (which escapes
            # them to `\\u003c`) or a plain `json.dumps`. transformers overrides the filter with
            # `json.dumps(..., ensure_ascii=False)`, which escapes neither -- a property of the
            # environment rather than of the template, so it is pinned rather than assumed.
            "description": "Current <conditions> & 'forecast'. Café.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "pattern": "^[A-Z].*$"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {"name": "close_issue", "description": "Close it.", "parameters": {}},
    },
]

# Every argument kind the parameter loop branches on. A string is emitted RAW (unquoted) and
# everything else through `tojson`; that is the one branch in `<parameter=...>` and a port that
# quoted strings would put `"Oslo"` where the model was trained on `Oslo`.
#
# The floats are the rows where `python_json` (serde_json with Python's separators) AGREES with
# `json.dumps`, measured by `v4_encoding::tests::boundary::numeric_rendering_diverges_from_python`
# and re-confirmed by this fixture: 0.5, 2.0, 0.1, 1e16 -> `1e+16`, 1e30 -> `1e+30`,
# 1e-4 -> `0.0001`. The DIVERGING rows (1e-5 and below, and parse-precision extremes) are
# deliberately absent -- pinning them would land a red, and the divergence is already gated in
# that file rather than half-fixed here.
ARG_SHAPES = {
    "s": "raw string, unquoted",
    "t": True,
    "f": False,
    "n": None,
    "i": 42,
    "neg": -7,
    "lst": [1, "two", None],
    "obj": {"k": "v", "n": 3},
    "f_half": 0.5,
    "f_two": 2.0,
    "f_tenth": 0.1,
    "f_e16": 1e16,
    "f_e30": 1e30,
    "f_em4": 1e-4,
}


def call(name, arguments=None, envelope=True):
    """One `tool_calls` entry, with or without the OpenAI `function` envelope."""
    fn = {"name": name}
    if arguments is not None:
        fn["arguments"] = arguments
    return {"id": "c1", "type": "function", "function": fn} if envelope else fn


def assistant(content=None, reasoning=None, calls=None):
    m = {"role": "assistant"}
    if content is not None:
        m["content"] = content
    if reasoning is not None:
        m["reasoning_content"] = reasoning
    if calls is not None:
        m["tool_calls"] = calls
    return m


# A history with two assistant turns straddling the last user query, which is the ONLY shape
# that can tell `preserve_thinking`'s three-part condition apart: the first assistant turn is
# before `last_query_index` and loses its think block, the second is after it and keeps it.
STRADDLE = [
    U,
    assistant(content="A1", reasoning="R1"),
    {"role": "user", "content": "Two"},
    assistant(content="A2", reasoning="R2"),
]

CASES = [
    # -- the thinking axis: FOUR states, because Jinja's `is true`/`is false` are IDENTITY -----
    # `enable_thinking = 1 / 0 / null / "true"` is none of undefined, true or false, so it lands
    # in a state neither branch was written for: NO reasoning instructions (so no system turn at
    # all) but an OPEN `<think>`. Collapsing this to a bool gets the SYSTEM prompt wrong, which
    # is Glimmer's `end_turn` scar (10 of 12 shapes wrong) on a different template.
    ("default_bare", {"messages": [U]}),
    ("default_gen_prompt", {"messages": [U], "add_generation_prompt": True}),
    ("thinking_true", {"messages": [U], "enable_thinking": True, "add_generation_prompt": True}),
    ("thinking_false", {"messages": [U], "enable_thinking": False, "add_generation_prompt": True}),
    ("thinking_false_no_gen_prompt", {"messages": [U], "enable_thinking": False}),
    ("thinking_null", {"messages": [U], "enable_thinking": None, "add_generation_prompt": True}),
    ("thinking_one", {"messages": [U], "enable_thinking": 1, "add_generation_prompt": True}),
    ("thinking_str_true", {"messages": [U], "enable_thinking": "true", "add_generation_prompt": True}),
    # -- the reasoning_effort axis, which the plan's ~31 cases did not anticipate --------------
    # THREE legal values and only TWO carry a sentence: `xhigh` is the DEFAULT and `medium` is
    # the quiet one. So the bare render above already carries a synthesised system turn, and
    # `medium` is the only legal value that suppresses it.
    ("effort_xhigh", {"messages": [U], "reasoning_effort": "xhigh"}),
    ("effort_medium", {"messages": [U], "reasoning_effort": "medium"}),
    ("effort_low", {"messages": [U], "reasoning_effort": "low"}),
    ("effort_low_gen_prompt", {"messages": [U], "reasoning_effort": "low", "add_generation_prompt": True}),
    # The whole effort block is gated on thinking, so with thinking OFF a bogus effort is never
    # resolved and never validated -- it does NOT refuse.
    ("effort_bogus_is_inert_when_thinking_off",
     {"messages": [U], "enable_thinking": False, "reasoning_effort": "not-a-level"}),
    # -- system turn shapes --------------------------------------------------------------------
    ("system_turn", {"messages": [{"role": "system", "content": "Be terse."}, U]}),
    ("system_turn_medium", {"messages": [{"role": "system", "content": "Be terse."}, U],
                            "reasoning_effort": "medium"}),
    ("system_turn_low", {"messages": [{"role": "system", "content": "Be terse."}, U],
                         "reasoning_effort": "low"}),
    # Empty system content: the instructions alone, with no `\n\n` separator.
    ("system_empty", {"messages": [{"role": "system", "content": ""}, U]}),
    # Empty system content AND no instructions: NOTHING is emitted. A port that opened a system
    # turn anyway would put an empty one in front of every request.
    ("system_empty_medium", {"messages": [{"role": "system", "content": ""}, U],
                             "reasoning_effort": "medium"}),
    # Whitespace-only system content trims to empty and takes the same branch.
    ("system_whitespace_only", {"messages": [{"role": "system", "content": "  \n\t "}, U],
                                "reasoning_effort": "medium"}),
    ("system_parts", {"messages": [{"role": "system", "content": [{"type": "text", "text": "Be terse."}]}, U]}),
    # -- conversation history and preserve_thinking --------------------------------------------
    ("multi_turn", {"messages": [U, assistant(content="Two"), {"role": "user", "content": "Three"}],
                    "add_generation_prompt": True}),
    ("assistant_reasoning", {"messages": [U, assistant(content="Short.", reasoning="Because.")]}),
    # The think block is emitted even when there is no reasoning: `<think>\n\n</think>\n\n`.
    ("assistant_no_reasoning", {"messages": [U, assistant(content="Short.")]}),
    # `reasoning_content is string` -- a non-string is treated as absent, not stringified.
    ("assistant_reasoning_not_a_string", {"messages": [U, assistant(content="A", reasoning=42)]}),
    ("assistant_reasoning_whitespace", {"messages": [U, assistant(content="A", reasoning="  R  \n")]}),
    ("preserve_thinking_undefined", {"messages": STRADDLE}),
    ("preserve_thinking_true", {"messages": STRADDLE, "preserve_thinking": True}),
    ("preserve_thinking_false", {"messages": STRADDLE, "preserve_thinking": False}),
    # `preserve_thinking = 1` is neither undefined nor `is true`, so it behaves like False --
    # the fourth state again, on the second kwarg that asks the same question.
    ("preserve_thinking_one", {"messages": STRADDLE, "preserve_thinking": 1}),
    ("preserve_thinking_null", {"messages": STRADDLE, "preserve_thinking": None}),
    # A user turn whose whole trimmed content is one `<tool_response>...</tool_response>` wrapper
    # is NOT a query, so `last_query_index` skips past it and the assistant turn after it keeps
    # its think block even with preserve off. This is the rule that makes a tool loop work.
    ("tool_response_user_turn_is_not_a_query", {"messages": [
        U, assistant(content="A1", reasoning="R1"),
        {"role": "user", "content": "<tool_response>\nr\n</tool_response>"},
        assistant(content="A2", reasoning="R2"),
    ], "preserve_thinking": False}),
    # -- tools: the system block, and its interaction with both other axes ---------------------
    ("tools_defs", {"messages": [U], "tools": TOOLS}),
    ("tools_defs_gen_prompt", {"messages": [U], "tools": TOOLS, "add_generation_prompt": True}),
    ("tools_medium", {"messages": [U], "tools": TOOLS, "reasoning_effort": "medium"}),
    ("tools_low", {"messages": [U], "tools": TOOLS, "reasoning_effort": "low"}),
    # With tools the system CONTENT is appended after the protocol block, separated by `\n\n` --
    # a different arrangement from the no-tools branch, which puts it after the instructions.
    ("tools_and_system", {"messages": [{"role": "system", "content": "Be terse."}, U], "tools": TOOLS}),
    ("tools_and_empty_system", {"messages": [{"role": "system", "content": ""}, U], "tools": TOOLS}),
    ("tools_thinking_off", {"messages": [U], "tools": TOOLS, "enable_thinking": False}),
    # `{%- if tools and ... %}` is PYTHON truthiness: an empty array and null are both FALSE and
    # the whole protocol block is omitted. Glimmer's port gated on Rust `Option`-ness and emitted
    # 1277 bytes of preamble for these; this preamble is larger.
    ("tools_empty", {"messages": [U], "tools": []}),
    ("tools_null", {"messages": [U], "tools": None}),
    # -- tool calls ----------------------------------------------------------------------------
    ("tool_call", {"messages": [U, assistant(calls=[call("get_weather", {"city": "Oslo"})])]}),
    # Content AND a call: the first call takes a `\n\n` lead. With empty content it takes none.
    ("tool_call_with_content",
     {"messages": [U, assistant(content="Let me look.", calls=[call("get_weather", {"city": "Oslo"})])]}),
    ("tool_call_two", {"messages": [U, assistant(calls=[
        call("get_weather", {"city": "Oslo"}), call("close_issue", {"n": 7})])]}),
    ("tool_call_arg_shapes", {"messages": [U, assistant(calls=[call("get_weather", ARG_SHAPES)])]}),
    # `arguments is defined and arguments != ''` -- both of these emit a call with no parameters.
    ("tool_call_no_arguments", {"messages": [U, assistant(calls=[call("get_weather")])]}),
    ("tool_call_empty_arguments", {"messages": [U, assistant(calls=[call("get_weather", "")])]}),
    ("tool_call_bare_function",
     {"messages": [U, assistant(calls=[call("get_weather", {"city": "Oslo"}, envelope=False)])]}),
    ("tool_call_and_tools", {"messages": [U, assistant(calls=[call("get_weather", {"city": "Oslo"})])],
                             "tools": TOOLS}),
    # The OBJECT form of the call that `a_json_string_arguments_renders_as_the_object_the_reference_was_given`
    # scores the port's JSON-string divergence against. The reference RAISES on the string form
    # (`items` on a `str`), so this is the only reference bytes that divergence can be pinned to.
    ("tool_call_args_object_for_the_string_divergence",
     {"messages": [U, assistant(calls=[call("get_weather", {"city": "Oslo", "days": 3})])]}),
    # -- tool results, which the template frames as USER turns ---------------------------------
    ("tool_result", {"messages": [U, assistant(calls=[call("get_weather", {"city": "Oslo"})]),
                                  {"role": "tool", "content": "4C, rain"}]}),
    # Consecutive results share ONE `<|im_start|>user` block: the opener appears only when the
    # previous message was not a tool, the closer only when the next one is not.
    ("tool_result_two", {"messages": [U, assistant(content="x"),
                                      {"role": "tool", "content": "a"},
                                      {"role": "tool", "content": "b"}]}),
    ("tool_result_then_user", {"messages": [U, {"role": "tool", "content": "a"},
                                            {"role": "user", "content": "and?"}]}),
    # A conversation OPENING with a tool message gets NO opener -- `loop.previtem` is undefined
    # on the first iteration, which is falsy. Reachable, because the user-query scan only needs a
    # real user turn to exist somewhere.
    ("tool_result_first", {"messages": [{"role": "tool", "content": "a"}, U]}),
    ("tool_result_parts", {"messages": [U, {"role": "tool", "content": [{"type": "text", "text": "4C"}]}]}),
    # -- content shapes ------------------------------------------------------------------------
    ("content_parts_text", {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "one "}, {"type": "text", "text": "two"}]}]}),
    ("content_null", {"messages": [{"role": "user", "content": None}]}),
    ("content_missing", {"messages": [{"role": "user"}]}),
    ("content_empty_parts", {"messages": [{"role": "user", "content": []}]}),
    # Jinja's `trim` is Python's `str.strip()`, which strips U+001C-U+001F as well as Unicode
    # whitespace -- Rust's `str::trim()` does NOT, because those four are not in the White_Space
    # property. NBSP, U+2000 and the ideographic space ARE stripped by both; U+200B by neither.
    # All of that is in this one case, so the port's `jinja_trim` is scored rather than argued.
    ("whitespace_trim", {"messages": [{"role": "user",
        "content": "\x1c\x1d\x1e\x1f \t\n\xa0 　  Hi​ there  　 \xa0\n\t \x1f\x1e\x1d\x1c"}]}),
    ("whitespace_only_user", {"messages": [{"role": "user", "content": "   \n  "}]}),
    # -- vision placeholders. Text-only is v1's scope, but the FRAMING is this module's job and
    #    the counters are render-wide state the template keeps in a `namespace()`.
    ("content_parts_image", {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "look: "}, {"type": "image"}]}]}),
    ("content_parts_video", {"messages": [{"role": "user", "content": [{"type": "video"}]}]}),
    # `'image' in item` is KEY membership, so `{"image_url": ...}` is an image by its key alone
    # and never reaches the text arm even when it also carries `text`.
    ("content_parts_image_url_key", {"messages": [{"role": "user", "content": [
        {"image_url": "http://x", "text": "ignored"}]}]}),
    # The counters number across the whole render, in main-loop message order.
    ("vision_ids", {"messages": [
        {"role": "user", "content": [{"type": "image"}, {"type": "video"}]},
        assistant(content="ok"),
        {"role": "user", "content": [{"type": "image"}, {"type": "video"}]},
    ], "add_vision_id": True}),
    ("vision_ids_off", {"messages": [
        {"role": "user", "content": [{"type": "image"}, {"type": "video"}]},
        assistant(content="ok"),
        {"role": "user", "content": [{"type": "image"}, {"type": "video"}]},
    ]}),
    # -- the REJECT direction: every `raise_exception` the template can reach through this door --
    ("refuse_unknown_role", {"messages": [{"role": "developer", "content": "d"}, U]}),
    ("refuse_no_user_query", {"messages": [{"role": "system", "content": "S"}]}),
    ("refuse_no_user_query_assistant_only", {"messages": [assistant(content="A")]}),
    # Every user turn is a tool_response wrapper, so none of them is a query.
    ("refuse_no_user_query_all_tool_responses",
     {"messages": [{"role": "user", "content": "<tool_response>\nr\n</tool_response>"}]}),
    ("refuse_system_not_first", {"messages": [U, {"role": "system", "content": "S"}]}),
    ("refuse_system_image", {"messages": [{"role": "system", "content": [{"type": "image"}]}, U]}),
    ("refuse_system_video", {"messages": [{"role": "system", "content": [{"type": "video"}]}, U]}),
    ("refuse_unexpected_part", {"messages": [{"role": "user", "content": [{"foo": "bar"}]}]}),
    ("refuse_unexpected_content_type", {"messages": [{"role": "user", "content": 42}]}),
    ("refuse_bad_effort", {"messages": [U], "reasoning_effort": "high"}),
    # Jinja's `default` filter replaces UNDEFINED only, so a null reaches the membership test as
    # `None` and is refused with `None` in the message -- NOT defaulted to xhigh. Reading this as
    # "absent" is the `Option::is_some` mistake on a kwarg that decides the system prompt.
    ("refuse_effort_null", {"messages": [U], "reasoning_effort": None}),
    ("refuse_effort_empty_string", {"messages": [U], "reasoning_effort": ""}),
    # A content part error in a LATER message refuses before the misplaced-system check, because
    # the template renders content at the top of the loop body and checks the role after it.
    # Error ORDERING is part of the port.
    ("refuse_content_before_role", {"messages": [U, {"role": "nope", "content": 42}]}),
]


def main() -> int:
    ckpt = os.environ.get("QWEN_CKPT")
    if not ckpt:
        print("QWEN_CKPT must point at the checkpoint directory", file=sys.stderr)
        return 2
    import transformers  # noqa: PLC0415 -- import after the arg check
    import jinja2  # noqa: PLC0415
    import tokenizers  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415

    tk = AutoTokenizer.from_pretrained(ckpt)
    out, refused = [], 0
    for name, kwargs in CASES:
        kw = dict(kwargs)
        case = {"name": name, "kwargs": kw}
        try:
            text = tk.apply_chat_template(
                kw["messages"],
                tokenize=False,
                **{k: v for k, v in kw.items() if k != "messages"},
            )
        except Exception as exc:  # noqa: BLE001 -- the message IS the pinned value
            # Only the template's own `raise_exception` is a contract. A transformers-level
            # refusal would mean the template was never entered, and asserting the port
            # reproduces a message from outside the template is a claim about the wrong file.
            kind = type(exc).__name__
            if kind not in ("TemplateError", "TypeError"):
                print(f"  {name}: refused UPSTREAM by transformers ({kind}), not pinnable: {exc}",
                      file=sys.stderr)
                return 3
            case["raises"] = str(exc)
            case["raises_type"] = kind
            refused += 1
            out.append(case)
            continue
        # The ids are pinned too, and for a reason the string cannot cover: the port renders
        # TEXT, and text is only useful if the tokenizer resolves `<|im_start|>` and `<think>`
        # to ONE id each rather than to their constituent pieces. A port that emitted a
        # lookalike would match the string byte-for-byte and tokenize differently.
        case["expected"] = text
        case["ids"] = tk.encode(text, add_special_tokens=False)
        out.append(case)
    doc = {
        "provenance": {
            "repo": REPO,
            "revision": REVISION,
            "chat_template.jinja": {"bytes": TEMPLATE_BYTES, "sha256": TEMPLATE_SHA256,
                                    "fnv1a64": TEMPLATE_FNV1A64},
            "tokenizer.json": {"sha256": TOKENIZER_SHA256},
            "tokenizer_config.json": {"sha256": TOKENIZER_CONFIG_SHA256},
            "python": sys.version.split()[0],
            "transformers": transformers.__version__,
            "jinja2": jinja2.__version__,
            "tokenizers": tokenizers.__version__,
        },
        "cases": out,
    }
    dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen-chat-cases.json")
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    print(f"qwen_template_driver: {len(out)} cases ({refused} refusals) -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
