"""Run the checkpoint's own `encoding_dsv4.py` over a batch of cases.

Driven by `tests/v4_encoding.rs::differential_against_the_reference`, which generates the
cases, calls this, and diffs. Kept as a file rather than a string inside the Rust so it can
be run by hand while debugging a divergence — the Rust leaves the corpus behind when it
fails, so the path it printed can be fed straight back in:

    python3 tests/dsv4_reference_driver.py <encoding-dir> <cases.json>

`cases.json` is `[{messages, thinking, effort, drop, bos}, ...]`. Emits, to stdout, one
object per case: `{"encoded": <str|null>, "parses": [<parse|null>, ...]}`. `encoded` is null
where the reference raised. `turns` are the assistant spans it sliced (the Rust asserts its own
slicing against them) and `parses` runs the reference's OWN parser back over each, so the differential covers both directions rather than
only encode — a parser agreeing with a wrong encoder is exactly the failure a round-trip
catches and two hand-written cases do not.
"""

import json
import sys

encoding_dir, cases_path = sys.argv[1:3]
sys.path.insert(0, encoding_dir)
from encoding_dsv4 import (  # noqa: E402  (path set above)
    encode_messages,
    parse_message_from_completion_text,
)

MARKER = "<｜Assistant｜><think>"
USER = "<｜User｜>"


def assistant_turns(prompt):
    """Every completion-shaped span: after a thinking-mode assistant prefix, up to the next
    user turn or the end. One expression, mirroring `assistant_turns` in
    `tests/v4_encoding.rs` — they must slice identically or a mismatch is the slicing's
    fault rather than the parser's, and two one-liners can be compared at a glance."""
    return [s.split(USER, 1)[0] for s in prompt.split(MARKER)[1:]]


def parse(turn):
    try:
        p = parse_message_from_completion_text(turn, "thinking")
    except Exception:
        # A refusal is a legitimate outcome — a trailing generation prompt and a `wo_eos`
        # turn are both unparseable by design — and the Rust must refuse the same span.
        return None
    return {
        "content": p["content"],
        "reasoning": p["reasoning_content"],
        "calls": [[c["function"]["name"], c["function"]["arguments"]] for c in p["tool_calls"]],
    }


with open(cases_path, encoding="utf-8") as f:
    cases = json.load(f)

out = []
for case in cases:
    # Read the case OUTSIDE the try. Inside it, a key the Rust renamed would be caught by the
    # `except` and recorded as "the reference refused", turning a broken harness into a green
    # run.
    messages = case["messages"]
    thinking, effort = case["thinking"], case["effort"]
    drop, bos = case["drop"], case["bos"]
    try:
        encoded = encode_messages(
            messages,
            thinking_mode=thinking,
            drop_thinking=drop,
            add_default_bos_token=bos,
            reasoning_effort=effort,
        )
    except Exception:
        out.append({"encoded": None, "turns": [], "parses": []})
        continue
    turns = assistant_turns(encoded)
    # `turns` is emitted, not just the parses: the Rust re-slices the same string and asserts
    # its spans equal these. Comparing only the COUNT let a boundary divergence hide inside the
    # 207-of-282 turns that both sides refuse, where null == Err regardless of what was sliced.
    out.append({"encoded": encoded, "turns": turns, "parses": [parse(t) for t in turns]})

# Straight to the byte stream with an explicit encoding. `json.dump(..., sys.stdout)` would
# take the interpreter's stdout encoding, and this corpus carries CJK and an emoji — under a
# non-UTF-8 locale that raises UnicodeEncodeError and the Rust reports it as a driver crash
# rather than as what it is.
sys.stdout.buffer.write(json.dumps(out, ensure_ascii=False).encode("utf-8"))
