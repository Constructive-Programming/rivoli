#!/usr/bin/env python3
"""Vendor Muse Glimmer's OWN chat template output, so the Rust hand-port is pinned against
the real renderer rather than against a second transcription.

`docs/investigations/glimmer-integration.md` S4 item 2 asks for the template "hand-ported and
byte-pinned". This is the *pin*: it renders the checkpoint's `chat_template.jinja` through
`AutoTokenizer.apply_chat_template` -- the same Jinja sandbox, the same macros, the same
`tojson` -- and writes `(name, kwargs, expected)` triples to `tests/glimmer-chat-cases.json`.
`tests/glimmer_template.rs` reads them with no network, no GPU and no Python.

**Why not pin against a literal written by hand.** GLM's template drifted for months to
GLM-4's `<|role|>\\n` framing (memory: `artifact-drops-the-chat-template`), and the reason it
went unnoticed is that the only thing it was ever checked against was a second reading by the
same author. That is the same failure `glimmer_chain.rs` carries as a caveat and
`glimmer_reference.rs` closes. A pin whose expected side comes from the model's own file
cannot share a misreading with the port.

**Every case fixes `current_date`.** The template reads `current_date` if defined and
otherwise calls `strftime_now('%Y-%m-%d')`, so an unpinned render is a fixture that goes red
at midnight. The Rust side takes the date as a parameter for the same reason -- that is a
property of the template, not a convenience.

Usage (no GPU, no lock, no wrapper script -- it is one command and takes seconds):

    HF_HUB_OFFLINE=1 GLIMMER_CKPT=/swarm/storage/ai/rivoli/muse-glimmer-30b \\
    /home/rhansen/glimmer-anchor/venv/bin/python tests/glimmer_template_driver.py

The checkpoint directory needs only `tokenizer.json`, `tokenizer_config.json` and
`chat_template.jinja` -- about 27 MB of a 59.582 GB repo, so regenerating this does not need
the weights. `HF_HUB_OFFLINE=1` keeps `from_pretrained` from reaching for the hub when the
local directory is a partial clone, which it is while the shards are still downloading.
"""

import json
import os
import sys

# The date every case pins. Any value works; what matters is that it is a CONSTANT, so the
# vendored bytes do not depend on when the driver ran. Chosen as the day the pin was made.
DATE = "2026-08-14"

# A tool set that exercises both namespacing branches at once: `github.*` collapses two
# functions into one namespace line, and the bare `get_weather` shows what a function with no
# namespace does to the recipient list (`"get_weather.*"` -- the template splits on '.' and
# takes element 0 whether or not there is a dot, which is worth pinning rather than assuming).
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "github.create_issue",
            "description": "Open an issue.",
            "parameters": {
                "type": "object",
                "properties": {"title": {"type": "string"}},
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github.close_issue",
            "description": "Close an issue.",
            "parameters": {"type": "object", "properties": {"n": {"type": "integer"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            # The same `tojson` question as ATEM_ARGS["escapes"], asked on the OTHER side of
            # the template: the tool-definitions block renders name/description/parameters
            # through it, and a schema with `<` in a description or a `pattern` is ordinary.
            "description": "Current <conditions> & 'forecast'.",
            "parameters": {"type": "object", "properties": {
                "city": {"type": "string", "pattern": "^[A-Z].*$", "description": 'a "place"'},
            }},
        },
    },
]

# Every argument kind `render_atem` branches on, in one call: string (raw), bool (true/false
# with the template's own literal), none (`null`), list and mapping (both `| tojson`), and a
# number (the else arm). The bool arms in particular are written as bare `true`/`false` text
# inside `{%- if -%}` blocks in the Jinja, which is exactly the shape a hand-port gets wrong.
ATEM_ARGS = {
    "title": "a string, unquoted",
    "flag_t": True,
    "flag_f": False,
    "nothing": None,
    "list": [1, "two", None],
    "map": {"k": "v", "n": 3},
    "count": 42,
    # Floats go down the `else` arm as Jinja's `{{ v }}`, i.e. Python `str()`. Pinned because
    # Rust's `f64` Display and Python's `str` disagree on several shapes (`1e30`, `1.0`), and
    # a port that reached for `to_string()` would be right on `1.5` and wrong on the rest.
    "ratio": 1.5,
    "whole": 2.0,
    "huge": 1e30,
    # BOTH boundaries of Python's fixed-vs-scientific rule, on both sides. Added after a red
    # proof showed the port's threshold constant could be moved from 16 to 17 with every case
    # still green: `1e30` is far past the boundary and `2.0` far short of it, so nothing in the
    # fixture was near enough to notice. A constant no case straddles is a constant nothing
    # checks.
    "hi_fixed": 1e15,
    "hi_sci": 1e16,
    "lo_fixed": 1e-4,
    "lo_sci": 1e-5,
    # `<` `>` `&` `'` decide whether `tojson` is Jinja's HTML-safe one (which escapes them to
    # `\\u003c` and friends) or a plain `json.dumps`. The answer is not guessable from the
    # template -- it is a property of the environment transformers compiles it in -- and it
    # changes the bytes the model was trained on, so it is pinned rather than assumed.
    "escapes": ["<a>", "b & c", "it's", 'say "hi"', "café"],
}

CASES = [
    # -- the bounded-greedy-run path, and the one every decode uses --------------------
    ("plain_user", {"messages": [{"role": "user", "content": "Hi"}], "add_generation_prompt": True}),
    ("no_generation_prompt", {"messages": [{"role": "user", "content": "Hi"}]}),
    # The synthesised system block's three kwargs.
    ("reasoning_low", {"messages": [{"role": "user", "content": "Hi"}], "reasoning_strength": "low", "add_generation_prompt": True}),
    ("knowledge_cutoff", {"messages": [{"role": "user", "content": "Hi"}], "knowledge_cutoff": "2025-12-01", "add_generation_prompt": True}),
    # -- an explicit system turn suppresses the synthesised one ------------------------
    ("explicit_system", {"messages": [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "Hi"}], "add_generation_prompt": True}),
    # The four `Reasoning effort` casings the template normalises, and the suppression that
    # follows: a system prompt that already carries the directive must NOT get a second one.
    ("system_effort_lower", {"messages": [{"role": "system", "content": "reasoning effort: low"}, {"role": "user", "content": "Hi"}], "add_generation_prompt": True}),
    ("system_effort_title", {"messages": [{"role": "system", "content": "Reasoning Effort: Low"}, {"role": "user", "content": "Hi"}], "add_generation_prompt": True}),
    ("system_effort_upper", {"messages": [{"role": "system", "content": "REASONING EFFORT: LOW"}, {"role": "user", "content": "Hi"}], "add_generation_prompt": True}),
    ("system_effort_sentence", {"messages": [{"role": "system", "content": "Reasoning effort: low"}, {"role": "user", "content": "Hi"}], "add_generation_prompt": True}),
    # -- conversation history ----------------------------------------------------------
    ("multi_turn", {"messages": [
        {"role": "user", "content": "One"},
        {"role": "assistant", "content": "Two"},
        {"role": "user", "content": "Three"},
    ], "add_generation_prompt": True}),
    ("assistant_reasoning", {"messages": [
        {"role": "user", "content": "Why?"},
        {"role": "assistant", "reasoning_content": "Because.", "content": "Short answer."},
    ], "add_generation_prompt": True}),
    ("assistant_recipient", {"messages": [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "recipient": "self", "content": "thinking out loud"},
    ], "add_generation_prompt": True}),
    ("assistant_end_turn_false", {"messages": [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "part one", "end_turn": False},
    ], "add_generation_prompt": True}),
    # -- multimodal content parts, which is how `<|patch|>`/`<|video|>` reach the text ---
    ("content_parts", {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "look: "},
        {"type": "image"},
        {"type": "text", "text": " and "},
        {"type": "video"},
    ]}], "add_generation_prompt": True}),
    ("content_none", {"messages": [{"role": "user", "content": None}], "add_generation_prompt": True}),
    # -- tools: the definitions block, the ATEM call markup, and the result turn ---------
    ("tools_defs", {"messages": [{"role": "user", "content": "Hi"}], "tools": TOOLS, "add_generation_prompt": True}),
    ("tools_namespace_descriptions", {"messages": [{"role": "user", "content": "Hi"}], "tools": TOOLS,
                                      "tool_namespace_descriptions": {"github": "The GitHub API."}, "add_generation_prompt": True}),
    ("tool_call", {"messages": [
        {"role": "user", "content": "Open one"},
        {"role": "assistant", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "github.create_issue", "arguments": ATEM_ARGS}},
        ]},
    ], "tools": TOOLS, "add_generation_prompt": True}),
    ("tool_call_two", {"messages": [
        {"role": "user", "content": "Open two"},
        {"role": "assistant", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "github.create_issue", "arguments": {"title": "a"}}},
            {"id": "c2", "type": "function", "function": {"name": "github.close_issue", "arguments": {"n": 7}}},
        ]},
    ], "tools": TOOLS, "add_generation_prompt": True}),
    ("tool_result_named", {"messages": [
        {"role": "user", "content": "Weather?"},
        {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Oslo"}}}]},
        {"role": "tool", "name": "get_weather", "content": "4C, rain"},
    ], "tools": TOOLS, "add_generation_prompt": True}),
    # No `name` -- the template walks every earlier message's tool_calls to recover it by id.
    ("tool_result_by_id", {"messages": [
        {"role": "user", "content": "Weather?"},
        {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Oslo"}}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "4C, rain"},
    ], "tools": TOOLS, "add_generation_prompt": True}),
    # Neither name nor a resolvable id: the namespace falls back to the id, then to ''.
    ("tool_result_unresolved", {"messages": [
        {"role": "user", "content": "Weather?"},
        {"role": "tool", "tool_call_id": "nosuch", "content": "4C, rain"},
    ], "add_generation_prompt": True}),
    # -- the branch nothing else reaches: `end_token` is only READ in the tool_calls arm,
    #    and only differs from `<|eot|>` when the NEXT message has the same role.
    ("tool_call_then_assistant", {"messages": [
        {"role": "user", "content": "Go"},
        {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Oslo"}}}]},
        {"role": "assistant", "content": "done"},
    ], "tools": TOOLS, "add_generation_prompt": True}),
    # An unknown role renders as NOTHING -- the template's `if/elif` chain has no else. Pinned
    # because a hand-port that framed it as `user` (which GLM's does, deliberately) would be
    # wrong HERE, and silently.
    ("unknown_role", {"messages": [
        {"role": "developer", "content": "dropped entirely"},
        {"role": "user", "content": "Hi"},
    ], "add_generation_prompt": True}),
]


def main() -> int:
    ckpt = os.environ.get("GLIMMER_CKPT")
    if not ckpt:
        print("GLIMMER_CKPT must point at the checkpoint directory", file=sys.stderr)
        return 2
    from transformers import AutoTokenizer  # noqa: PLC0415 -- import after the arg check

    tk = AutoTokenizer.from_pretrained(ckpt)
    out = []
    for name, kwargs in CASES:
        kw = dict(kwargs)
        kw.setdefault("current_date", DATE)
        text = tk.apply_chat_template(kw["messages"], tokenize=False,
                                      **{k: v for k, v in kw.items() if k != "messages"})
        # The ids are pinned too, and for a reason the string cannot cover: the port renders
        # text, and text is only useful if the tokenizer resolves `<|start|>` to ONE id rather
        # than to its five constituent pieces. A port that emitted a lookalike would match the
        # string byte-for-byte and tokenize differently.
        out.append({"name": name, "kwargs": kw, "expected": text,
                    "ids": tk.encode(text, add_special_tokens=False)})
    dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "glimmer-chat-cases.json")
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump({"source": "meta-models/Muse-Glimmer-30B chat_template.jinja",
                   "cases": out}, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    print(f"glimmer_template_driver: {len(out)} cases -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
