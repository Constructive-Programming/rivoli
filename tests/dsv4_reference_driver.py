"""Run the checkpoint's own `encoding_dsv4.py` over a batch of cases.

Driven by `tests/v4_encoding.rs::differential_against_the_reference`, which generates the
cases, calls this, and diffs. Kept as a file rather than a string inside the Rust so it can
be run by hand while debugging a divergence — the Rust leaves the corpus behind when it
fails, so the path it printed can be fed straight back in:

    python3 tests/dsv4_reference_driver.py <encoding-dir> <cases.json>

`cases.json` is `[{messages, thinking, effort, drop, bos}, ...]`; a JSON list of encoded
strings goes to stdout, with `null` wherever the reference raised.
"""

import json
import sys

encoding_dir, cases_path = sys.argv[1:3]
sys.path.insert(0, encoding_dir)
from encoding_dsv4 import encode_messages  # noqa: E402  (path set above)

with open(cases_path, encoding="utf-8") as f:
    cases = json.load(f)

out = []
for case in cases:
    # Read the case OUTSIDE the try. Inside it, a key the Rust renamed would be caught by
    # the `except` and recorded as "the reference refused", turning a broken harness into a
    # green run.
    messages = case["messages"]
    thinking, effort = case["thinking"], case["effort"]
    drop, bos = case["drop"], case["bos"]
    try:
        out.append(
            encode_messages(
                messages,
                thinking_mode=thinking,
                drop_thinking=drop,
                add_default_bos_token=bos,
                reasoning_effort=effort,
            )
        )
    except Exception:
        # A raise is a legitimate outcome and the Rust must refuse the same case; recorded
        # as null rather than aborting the batch.
        out.append(None)

# Straight to the byte stream with an explicit encoding. `json.dump(..., sys.stdout)` would
# take the interpreter's stdout encoding, and this corpus carries CJK and an emoji — under a
# non-UTF-8 locale that raises UnicodeEncodeError and the Rust reports it as a driver crash
# rather than as what it is.
sys.stdout.buffer.write(json.dumps(out, ensure_ascii=False).encode("utf-8"))
